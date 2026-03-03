import argparse
from math import sqrt
import numpy as np
from einops import einsum, rearrange
import torch
from torch import nn
from cs336_basics.module import TransformerLM
from cs336_basics.module import softmax
from cs336_basics.tokenizer import Tokenizer


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    tokens = rearrange(prompt, "s -> 1 s")
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(tokens)  # (1, seq_len, vocab_size)
        v = logits[:, -1, :]  # (1, vocab_size) — last position
        probs = softmax(v, dim=-1)  # (1, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return rearrange(tokens, "1 s -> s")


def main():
    p = argparse.ArgumentParser(
        description="Generate text from a trained TransformerLM"
    )
    p.add_argument("--ckpt", default="./checkpoints/ckpt_final.pt")
    p.add_argument("--vocab", default="./data/TinyStoriesV2-GPT4-train_BPE_vocab.pkl")
    p.add_argument("--merges", default="./data/TinyStoriesV2-GPT4-train_BPE_merges.pkl")
    p.add_argument(
        "--prompt",
        default="Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, “Wow, that is a really amazing vase! Can I buy it?” The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!” So Ben",
    )

    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        theta=10000.0,
        device=device,
    )
    ckpt_path = "./checkpoints/ckpt_final.pt"
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    tokenizer = Tokenizer.from_files(
        vocab_filepath="./data/TinyStoriesV2-GPT4-train_BPE_vocab.pkl",
        merges_filepath="./data/TinyStoriesV2-GPT4-train_BPE_merges.pkl",
    )

    prompt_text = "Once upon a time"
    prompt = torch.tensor(tokenizer.encode(prompt_text)).to(device)
    output = generate(model, prompt, 100)
    answer = tokenizer.decode(output.to("cpu").numpy())
    print(answer)
