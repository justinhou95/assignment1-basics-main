from math import sqrt

from einops import einsum, rearrange
import torch
from torch import nn


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # inputs: (..., vocab_size), targets: (...,)
    max_logits = inputs.max(dim=-1, keepdim=True).values  # (..., 1)
    shifted = inputs - max_logits  # subtract max for stability
    log_sum_exp = torch.log(
        torch.exp(shifted).sum(dim=-1)
    )  # (...,)  -- log cancels exp
    target_logits = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (...,)
    loss = -target_logits + max_logits.squeeze(-1) + log_sum_exp
    return loss.mean()


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return einsum(attn, V, "... q k, ... k d -> ... q d")


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... i, o i -> ... o")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        return self.w2(gate * torch.sigmoid(gate) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        positions = torch.arange(max_seq_len, device=device)  # (max_seq_len,)
        freqs = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device) / d_k)
        )  # (d_k//2,)
        angles = einsum(positions, freqs, "i, j -> i j")  # (max_seq_len, d_k//2)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k),  token_positions: (..., seq_len)
        cos = self.cos[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k//2)

        x_even = x[..., 0::2]  # (..., seq_len, d_k//2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k//2)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        # d_model = d_k * num_heads = d_v * num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, seq_len, _ = x.shape

        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: (batch, num_heads, seq_len, d_k)

        Q = rearrange(
            Q,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        K = rearrange(
            K,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        V = rearrange(
            V,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Causal mask: lower triangular, (seq_len, seq_len)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        # (batch, num_heads, seq_len, d_k)
        out = scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads: (batch, seq_len, d_model)
        out = rearrange(
            out, "batch num_heads seq_len d_k -> batch seq_len (num_heads d_k)"
        )
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.context_length = context_length

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embeddings(x)
        if token_positions is None:
            token_positions = torch.arange(x.shape[1], device=x.device)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=200, eos_id=None):
        tokens = rearrange(tokens, "s -> 1 s")
        self.eval()
        for _ in range(max_new_tokens):
            ctx = tokens[:, -self.context_length :]  # works for any prompt length
            logits = self.forward(ctx)  # (1, ctx_len, vocab_size)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_id is not None and next_token.item() == eos_id:
                break
        return rearrange(tokens, "1 s -> s")
