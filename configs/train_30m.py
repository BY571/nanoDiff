"""~30M total params (~11M non-embedding) — the smallest "real" config.

A serious small-model run on real text. 30k iters at 16K tokens/iter sees
~480M tokens, so prepare at least ~500M tokens of FineWeb-Edu:

    python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 500_000_000
    NANODIFF_WANDB=1 python train.py --config configs/train_30m.py
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-30m",

    # ---- model  (~11M non-embedding params) ----
    n_layer=6,
    n_head=6,
    n_embd=384,
    block_size=512,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization ----
    # effective batch = 8 * 4 = 32 sequences (~16K tokens / iter)
    batch_size=8,
    grad_accum_steps=4,
    max_iters=30_000,
    lr=6e-4,
    min_lr=1e-5,
    warmup_iters=1_000,
    decay_iters=5_000,

    # ---- io ----
    out_dir="checkpoints/30m",
)
