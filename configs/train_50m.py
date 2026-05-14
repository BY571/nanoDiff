"""~50M non-embedding parameters. The smallest "real" config — fast to iterate,
trains on a single modest GPU, good for debugging changes before scaling up.

    python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 1_000_000_000
    python train.py --config configs/train_50m.py
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-50m",

    # ---- model  (~49.5M non-embedding params) ----
    n_layer=7,
    n_head=12,
    n_embd=768,
    block_size=1024,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization ----
    # effective batch = 32 * 8 * 1 = 256 sequences (~256K tokens / iter)
    batch_size=32,
    grad_accum_steps=8,
    max_iters=50_000,
    lr=6e-4,
    min_lr=1e-5,
    warmup_iters=1500,
    decay_iters=10_000,

    # ---- io ----
    out_dir="checkpoints/50m",
)
