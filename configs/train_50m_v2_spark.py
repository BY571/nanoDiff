"""DGX-Spark · 50M v2 · new baseline at block_size=512.

Identical to `train_50m_spark.py` except `block_size: 1024 → 512`.

Motivation: a microbench on the Spark (see commit log around 2026-05-19)
showed the GPU is saturated at the original config — wall-clock per step is
proportional to (batch * block_size). Halving block_size halves time/step
(3640 → 1601 ms) for the same model and the same effective batch. We see
half the tokens per iter; at max_iters=16k that's ~1B tokens total — which
is Chinchilla-optimal for 50M (vs the original 2B, which was over-trained).

So this run is *faster* AND *more principled* than the original:
    wall-clock:    ~12 hr → ~7 hr
    tokens seen:   ~2.1B  → ~1.05B (Chinchilla-optimal for 50M)
    everything else (model, batch, LR, schedule, decay) is unchanged.

Once this lands and matches the old baseline within ~0.1 nats, it becomes
the new control for the schedule sweep (cosine, WSD-long-decay).

Run:
    NANODIFF_WANDB=1 python train.py --config configs/train_50m_v2_spark.py
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-50m-v2-spark",

    # ---- model (~11M non-embedding params under tied embeddings) ----
    n_layer=7,
    n_head=12,
    n_embd=768,
    block_size=512,   # <- halved from 1024

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization (unchanged from train_50m_spark.py) ----
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

    # ---- io (separate dir so the old block=1024 ckpt isn't overwritten) ----
    out_dir="checkpoints/50m_v2_spark",
)
