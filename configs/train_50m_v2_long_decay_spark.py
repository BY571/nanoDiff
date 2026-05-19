"""DGX-Spark · 50M v2 · WSD with long decay (decay_iters 3k -> 10k).

Schedule-sweep variant #2. Identical to `train_50m_v2_spark.py` except:
    decay_iters: 3_000 -> 10_000

Tests the *intermediate* hypothesis between WSD and cosine: keep the WSD
shape (stable + decay) but reallocate time from stable to decay.

    warmup 500  ->  stable 500..6000  ->  decay 6000..16000

If WSD-long-decay wins, the *holding* of peak LR has value but we just need
less of it. If cosine wins (no stable at all), the holding has no value.

Run:
    NANODIFF_WANDB=1 python train.py --config configs/train_50m_v2_long_decay_spark.py

Estimated wall-clock: ~5 hr.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-50m-v2-long-decay-spark",

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
    schedule="wsd",
    decay_iters=10_000,        # <-- 3k -> 10k; stable now only iters 500..6000

    # ---- evaluation ----
    eval_interval=500,

    # ---- system ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/50m_v2_long_decay_spark",
)
