"""DGX-Spark · 50M model · capacity sweep entry (run 1 of 3).

Part of the Spark capacity sweep — three runs (50M / 150M / 350M) on the same
2B-token FineWeb-Edu dataset, identical schedule and batch settings, only the
model architecture varying. Goal: a clean scaling curve at fixed data, directly
comparable to the 30M Run 3 from `EXPERIMENTS.md`.

Token budget per run: 16k iters × 131K tokens/iter = ~2.1B tokens (one epoch).

Data prep on Spark (do this once before the sweep):
    python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 2_000_000_000

Run:
    NANODIFF_WANDB=1 python train.py --config configs/train_50m_spark.py

Estimated wall-clock on DGX-Spark: ~2 hrs.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-50m-spark",

    # ---- model (~11M non-embedding params under tied embeddings) ----
    n_layer=7,
    n_head=12,
    n_embd=768,
    block_size=1024,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization (Spark-tuned) ----
    # Effective batch = 32 * 4 = 128 sequences = 131K tokens/iter.
    # batch_size=128, grad_accum=1 hit OOM at first training step because the
    # diffusion loss's logits.float() cast materializes a ~26 GB fp32 tensor
    # (B*T*V at B=128); kept here at 32 to keep the float-logits tensor ~6.6 GB.
    # 4× our Jetson effective batch (32), so LR is sqrt-scaled: 6e-4 → 1.2e-3.
    # 16k iters × 131K = ~2.1B tokens seen = ~1 epoch of the 2B dataset.
    batch_size=32,
    grad_accum_steps=4,
    max_iters=16_000,
    lr=1.2e-3,
    min_lr=1e-5,
    warmup_iters=500,
    decay_iters=3_000,

    # ---- evaluation cadence (32 evals over the run) ----
    eval_interval=500,

    # ---- system (Spark has working compile and bf16) ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/50m_spark",
)
