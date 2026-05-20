"""nanoDiff · 150M base model · pretraining config.

A masked diffusion LM with ~150M non-embedding parameters — the next rung up
from the 50M base.

Why this run exists: an earlier 150M trained on only 2B tokens *tied* the 50M
(both val ~3.93) — a data-starvation artifact, not a capacity ceiling. This
config is data-matched: ~3B tokens, roughly Chinchilla-optimal for 150M, on a
fresh 10B-token FineWeb-Edu shard. It is the honest test of "does capacity
help once the model is properly fed?"

From the repo root:
    python scripts/prepare_data.py --out-dir data/fineweb_edu_10b --num-tokens 10_000_000_000
    python pretrain/train.py --config pretrain/configs/150m.py

Estimated wall-clock on a single DGX-Spark: ~1.5-2 days.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-150m",

    # ---- model (~150M non-embedding params) ----
    n_layer=12,
    n_head=16,
    n_embd=1024,
    block_size=512,         # the validated efficient setting (~3x faster than 1024)

    # ---- data ----
    data_dir="data/fineweb_edu_10b",

    # ---- optimization ----
    # batch 128 x block 512 = 65.5K tokens/iter; 46k iters ~= 3.0B tokens seen,
    # ~Chinchilla-optimal (20 tokens/param) for a 150M model.
    batch_size=128,
    grad_accum_steps=1,
    max_iters=46_000,
    lr=8e-4,                # scaled down from the 50M's 1.2e-3 (3x larger model)
    min_lr=1e-5,
    warmup_iters=1_000,
    decay_iters=15_000,     # ~33% decay (the 20-60% range all worked in the sweep)

    # ---- evaluation ----
    eval_interval=1_000,

    # ---- system ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/150m",
)
