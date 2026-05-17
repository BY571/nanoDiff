"""DGX-Spark · 150M model · capacity sweep entry (run 2 of 3).

Identical batch + schedule + data to the 50M and 350M sweep configs — only
n_layer/n_embd/n_head differ. See `train_50m_spark.py` for full sweep context.

Run:
    NANODIFF_WANDB=1 python train.py --config configs/train_150m_spark.py

Estimated wall-clock on DGX-Spark: ~3 hrs.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-150m-spark",

    # ---- model (~150M non-embedding params) ----
    n_layer=12,
    n_head=16,
    n_embd=1024,
    block_size=1024,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization (identical to the rest of the sweep) ----
    batch_size=128,
    grad_accum_steps=1,
    max_iters=16_000,
    lr=1.2e-3,
    min_lr=1e-5,
    warmup_iters=500,
    decay_iters=3_000,

    # ---- evaluation ----
    eval_interval=500,

    # ---- system ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/150m_spark",
)
