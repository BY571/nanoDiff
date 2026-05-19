"""DGX-Spark · 350M model · capacity sweep entry (run 3 of 3).

The largest model in the sweep. Identical batch + schedule + data to the
50M and 150M sweep configs. ~24 layers @ 1024-dim = ~300M non-embedding,
~350M total under tied embeddings.

Run:
    NANODIFF_WANDB=1 python train.py --config configs/train_350m_spark.py

Estimated wall-clock on DGX-Spark: ~5-6 hrs.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-350m-spark",

    # ---- model (~300M non-embedding params; ~355M total with tied embeddings) ----
    n_layer=24,
    n_head=16,
    n_embd=1024,
    block_size=1024,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization (identical to the rest of the sweep) ----
    # Effective batch = 32 * 4 = 128 (matches 50M sweep). At depth=24 × dim=1024,
    # microbatch=128 would need ~270 GB peak (2x the 150M). microbatch=32 fits
    # comfortably with grad accumulation preserving the effective batch.
    batch_size=32,
    grad_accum_steps=4,
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
    out_dir="checkpoints/350m_spark",
)
