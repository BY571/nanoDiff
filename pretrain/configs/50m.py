"""nanoDiff · 50M base model · pretraining config.

A masked diffusion LM with ~50M non-embedding parameters (~88M total under
tied embeddings). This config trained the public base checkpoint
[Sebasdi/nanodiff-50m-base](https://huggingface.co/Sebasdi/nanodiff-50m-base).

From the repo root:
    python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 2_000_000_000
    python pretrain/train.py --config pretrain/configs/50m.py
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-50m",

    # ---- model (~50M non-embedding params under tied embeddings) ----
    n_layer=7,
    n_head=12,
    n_embd=768,
    block_size=1024,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization ----
    # Effective batch = 128 sequences = 131K tokens/iter; 16k iters is ~2.1B
    # tokens, roughly one epoch of a 2B-token dataset.
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
    out_dir="checkpoints/50m",
)
