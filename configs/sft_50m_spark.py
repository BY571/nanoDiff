"""DGX-Spark · SFT the 50M base model on Alpaca-cleaned.

Fine-tunes `checkpoints/50m_spark/ckpt_final.pt` (the val-3.92 base, also on
the Hub as Sebasdi/nanodiff-50m-base) into an instruction-follower.

Prereqs:
    python scripts/prepare_sft_data.py --out-dir data/alpaca_sft

Run:
    NANODIFF_WANDB=1 python train_sft.py --config configs/sft_50m_spark.py

The model architecture fields below MUST match the base checkpoint — SFT loads
its weights with strict=True, so a mismatch fails loudly and immediately.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-50m-sft-alpaca",

    # ---- model — must match the 50M base checkpoint ----
    n_layer=7,
    n_head=12,
    n_embd=768,
    block_size=1024,        # base was trained at 1024; SFT seqs (P+L=512) fit fine

    # ---- start from the pretrained base ----
    init_from="checkpoints/50m_spark/ckpt_final.pt",

    # ---- SFT data (from scripts/prepare_sft_data.py) ----
    data_dir="data/alpaca_sft",

    # ---- optimization — short run, low LR (fine-tuning, not pretraining) ----
    # ~51k examples / batch 64 ~= 800 iters/epoch; 2400 iters ~= 3 epochs.
    # NOTE: the SFT loss plateaus by ~iter 200 (see EXPERIMENTS.md Run 9) —
    # ~500 iters is ample. 2400 is kept only to match the published checkpoint.
    batch_size=64,
    grad_accum_steps=1,
    max_iters=2_400,
    lr=1e-4,                # ~10x below the base run's 1.2e-3
    min_lr=1e-5,
    schedule="cosine",      # smooth decay, standard for fine-tuning
    warmup_iters=100,

    # ---- evaluation / sampling ----
    eval_interval=200,
    eval_iters=50,
    sample_interval=400,

    # ---- system ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/50m_sft_alpaca",
)
