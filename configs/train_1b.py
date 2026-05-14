"""~1.3B parameters. Intended for multi-GPU (torchrun) or a DGX-Spark node.

    python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 20_000_000_000
    torchrun --standalone --nproc_per_node=8 train.py --config configs/train_1b.py

Scaling knobs: keep `batch_size` (per-device micro-batch) small enough to fit in
memory, then grow the effective batch via `grad_accum_steps` and the number of GPUs.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-1b",

    # ---- model ----
    n_layer=24,
    n_head=16,
    n_embd=2048,
    block_size=1024,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization ----
    # effective batch = 8 * 16 * 8 GPUs = 1024 sequences (~1M tokens / iter)
    batch_size=8,
    grad_accum_steps=16,
    max_iters=200_000,
    lr=3e-4,
    min_lr=1e-5,
    warmup_iters=4000,
    decay_iters=40_000,

    # ---- io ----
    out_dir="checkpoints/1b",
)
