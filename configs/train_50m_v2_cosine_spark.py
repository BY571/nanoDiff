"""DGX-Spark · 50M v2 · cosine schedule (no stable phase).

Schedule-sweep variant #1. Identical to `train_50m_v2_spark.py` except:
    schedule: "wsd" -> "cosine"

Tests the hypothesis that the long stable phase in WSD is wasted compute.
Cosine starts dropping LR immediately from peak — same start (1.2e-3), same
end (1e-5), but smooth monotonic descent over all 15.5k post-warmup iters.

If cosine matches or beats WSD on final val, the WSD "stable phase" is
empty calories at this scale.

Run:
    NANODIFF_WANDB=1 python train.py --config configs/train_50m_v2_cosine_spark.py

Estimated wall-clock: ~5 hr.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-50m-v2-cosine-spark",

    # ---- model ----
    n_layer=7, n_head=12, n_embd=768, block_size=512,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization ----
    batch_size=128,
    grad_accum_steps=1,
    max_iters=16_000,
    lr=1.2e-3,
    min_lr=1e-5,
    warmup_iters=500,
    schedule="cosine",         # <-- the only thing that differs from the v2 control
    decay_iters=3_000,         # ignored under cosine but kept valid for the assert

    # ---- evaluation ----
    eval_interval=500,

    # ---- system ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/50m_v2_cosine_spark",
)
