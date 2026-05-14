"""~150M non-embedding parameters. Fits comfortably on a single modern GPU.

    python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 2_000_000_000
    python train.py --config configs/train_150m.py
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-150m",

    # ---- model ----
    n_layer=12,
    n_head=16,
    n_embd=1024,
    block_size=1024,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization ----
    # effective batch = batch_size * grad_accum_steps * world_size
    #                 = 32 * 8 * 1  = 256 sequences  (~256K tokens / iter)
    batch_size=32,
    grad_accum_steps=8,
    max_iters=100_000,
    lr=4e-4,
    min_lr=1e-5,
    warmup_iters=2000,
    decay_iters=20_000,

    # ---- io ----
    out_dir="checkpoints/150m",
)
