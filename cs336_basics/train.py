"""
Training script for the CS336 Transformer language model.

Example usage:
    python -m cs336_basics.train \
        --train_path data/train.npy \
        --valid_path data/valid.npy \
        --vocab_size 50257 \
        --context_length 256 \
        --d_model 512 \
        --num_layers 6 \
        --num_heads 8 \
        --d_ff 2048 \
        --rope_theta 10000.0 \
        --batch_size 32 \
        --max_iters 10000 \
        --lr_max 3e-4 \
        --lr_min 1e-5 \
        --warmup_iters 500 \
        --weight_decay 0.1 \
        --grad_clip 1.0 \
        --log_interval 100 \
        --val_interval 500 \
        --checkpoint_dir checkpoints
"""

import argparse
import datetime
import os
import time

import numpy as np
import torch
import wandb
from einops import rearrange

from cs336_basics.module import TransformerLM, cross_entropy, softmax
from cs336_basics.optimizer import (
    AdamW,
    gradient_clipping,
    load_checkpoint,
    lr_cosine_schedule,
    save_checkpoint,
)


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    start: int | None = None,
):
    n = batch_size * context_length
    if start is None:
        start = np.random.randint(0, len(dataset) - n)
    inputs = torch.tensor(
        dataset[start : start + n].reshape(batch_size, context_length),
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        dataset[start + 1 : start + n + 1].reshape(batch_size, context_length),
        dtype=torch.long,
        device=device,
    )
    return inputs, targets


@torch.no_grad()
def estimate_val_loss(
    model, val_data, batch_size, context_length, device, num_batches=20
):
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(
            rearrange(logits, "b s v -> (b s) v"), rearrange(y, "b s -> (b s)")
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(args):
    device = args.device

    # ── Wandb ─────────────────────────────────────────────────────────────────
    wandb.init(project=args.wandb_project, config=vars(args))

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    run_name = f"{date_str}_{args.data}_{wandb.run.name}"
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_data = np.memmap(args.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.valid_path, dtype=np.uint16, mode="r")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from checkpoint '{args.resume}' at iteration {start_iter}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    t_start = t0

    for it in range(start_iter, args.max_iters):
        # Learning rate schedule
        lr = lr_cosine_schedule(
            it,
            max_learning_rate=args.lr_max,
            min_learning_rate=args.lr_min,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Forward + backward
        x, y = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device,
            start=0 if args.overfit_batch else None,
        )
        logits = model(x)
        loss = cross_entropy(
            rearrange(logits, "b s v -> (b s) v"), rearrange(y, "b s -> (b s)")
        )

        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if (it + 1) % args.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = (
                args.batch_size * args.context_length * args.log_interval / dt
            )
            print(
                f"iter {it+1:6d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{tokens_per_sec:,.0f} tok/s"
            )
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": it + 1,
                    "train/wallclock": time.time() - t_start,
                },
                step=it + 1,
            )
            t0 = time.time()

        # ── Validation ────────────────────────────────────────────────────────
        if (it + 1) % args.val_interval == 0:
            val_loss = estimate_val_loss(
                model, val_data, args.batch_size, args.context_length, device
            )
            print(f"  val loss: {val_loss:.4f}")
            wandb.log(
                {"val/loss": val_loss, "train/wallclock": time.time() - t_start},
                step=it + 1,
            )

            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{it+1:07d}.pt")
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            print(f"  checkpoint saved → {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"Training complete. Final checkpoint → {final_path}")
    wandb.finish()


DATA_CONFIGS = {
    "story": {
        "train_path": "./data/TinyStoriesV2-GPT4-train_tokens.bin",
        "valid_path": "./data/TinyStoriesV2-GPT4-valid_tokens.bin",
        "vocab_size": 10000,
    },
    "owt": {
        "train_path": "./data/owt_train_tokens.bin",
        "valid_path": "./data/owt_valid_tokens.bin",
        "vocab_size": 32000,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer LM")

    # Data
    p.add_argument(
        "--data",
        choices=["story", "owt"],
        default="story",
        help="Dataset to use: 'story' for TinyStoriesV2-GPT4, 'owt' for OpenWebText",
    )
    p.add_argument("--train_path", default=None, help="Override train data path")
    p.add_argument("--valid_path", default=None, help="Override valid data path")
    p.add_argument("--vocab_size", type=int, default=None)

    # Model
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimizer
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_iters", type=int, default=10000)
    p.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )

    # Logging & checkpointing
    p.add_argument("--wandb_project", type=str, default="cs336-lm")
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--val_interval", type=int, default=500)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    p.add_argument(
        "--overfit_batch",
        type=bool,
        default=False,
        help="Overfit to a single fixed batch (sanity check)",
    )

    args = p.parse_args()

    # Apply data config defaults (can be overridden by explicit flags)
    cfg = DATA_CONFIGS[args.data]
    if args.train_path is None:
        args.train_path = cfg["train_path"]
    if args.valid_path is None:
        args.valid_path = cfg["valid_path"]
    if args.vocab_size is None:
        args.vocab_size = cfg["vocab_size"]

    return args


if __name__ == "__main__":
    train(parse_args())
