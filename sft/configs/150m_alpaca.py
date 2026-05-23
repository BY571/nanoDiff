"""nanoDiff · 150M · SFT config — instruction-tune the 150M base on Alpaca-cleaned.

Fine-tunes the 150M base checkpoint into an instruction-follower using the LLaDA
Algorithm 2 recipe (response-only masking — see nanodiff/sft.py).

Mirrors sft/configs/50m_alpaca.py — same dataset, same schedule, same iter count,
so the 50M-SFT vs 150M-SFT comparison is methodologically clean. Only the model
arch (matched to the 150M base) and the LR (scaled with 1/12 of the larger
model's lower pretraining LR) differ.

From the repo root:
    python scripts/prepare_sft_data.py --out-dir data/alpaca_sft
    python sft/train.py --config sft/configs/150m_alpaca.py

The model architecture below MUST match the base checkpoint — SFT loads its
weights with strict=True, so a mismatch fails loudly and immediately.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-150m-sft-alpaca",

    # ---- model — must match the 150M base checkpoint ----
    n_layer=12,
    n_head=16,
    n_embd=1024,
    block_size=512,         # the 150M base was trained at 512 (not 1024 like the 50M)

    # ---- start from the pretrained 150M base ----
    init_from="checkpoints/150m/ckpt_final.pt",

    # ---- SFT data (already prepared by scripts/prepare_sft_data.py) ----
    data_dir="data/alpaca_sft",

    # ---- optimization — same shape as the 50M SFT, just rescaled LR ----
    # ~51k examples / batch 64 ≈ 800 iters/epoch; 5k iters ≈ 6 epochs.
    # The instruction *format* is learned within ~200 iters but the loss keeps
    # a slow real descent past that (the 2.4k-iter 50M run had not saturated).
    batch_size=64,
    grad_accum_steps=1,
    max_iters=5_000,
    lr=7e-5,                # 50M-SFT's 1e-4 × (150M's 8e-4 / 50M's 1.2e-3) ≈ 6.7e-5
    min_lr=1e-5,
    schedule="cosine",      # smooth decay, standard for fine-tuning
    warmup_iters=100,

    # ---- evaluation / sampling ----
    eval_interval=200,
    eval_iters=100,         # 100 (not 50) for a less noisy val curve
    sample_interval=500,

    # ---- system ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/150m_sft_alpaca",

    # ---- logging ----
    # SFT runs ~hours, not minutes — make wandb a config-level commitment
    # (same rule we just adopted for the 350M run).
    wandb_log=True,
)
