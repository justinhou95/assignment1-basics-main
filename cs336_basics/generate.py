import argparse

import torch
from einops import rearrange

from cs336_basics.module import TransformerLM, softmax
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
        probs = softmax(logits[:, -1, :], dim=-1)  # (1, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return rearrange(tokens, "1 s -> s")


def main():
    p = argparse.ArgumentParser(
        description="Generate text from a trained TransformerLM"
    )
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--vocab", required=True, help="Path to BPE vocab (.pkl)")
    p.add_argument("--merges", required=True, help="Path to BPE merges (.pkl)")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument(
        "--device",
        type=str,
        default=(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ),
    )
    args = p.parse_args()

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=args.device,
    )
    checkpoint = torch.load(args.ckpt, weights_only=True, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"],
    )
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), device=args.device)
    output_ids = generate(model, prompt_ids, args.max_new_tokens, eos_token_id=eos_id)
    print(tokenizer.decode(output_ids.cpu().tolist()))


if __name__ == "__main__":
    main()
