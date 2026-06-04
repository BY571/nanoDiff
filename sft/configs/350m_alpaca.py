"""nanoDiff · 350M · SFT config: instruction-tune the 350M base on Alpaca-cleaned.

Fine-tunes the 350M base checkpoint into an instruction-follower using the LLaDA
Algorithm 2 recipe (response-only masking, see nanodiff/sft.py).

Mirrors sft/configs/50m_alpaca.py and sft/configs/150m_alpaca.py exactly: same
dataset, schedule, batch, and iter count, so the 50M / 150M / 350M SFT
comparison stays methodologically clean. Only the model architecture and the
LR (scaled with 1/sqrt(width) from the 150M's 7e-5) differ.

From the repo root:
    python scripts/prepare_sft_data.py --out-dir data/alpaca_sft
    python sft/train.py --config sft/configs/350m_alpaca.py

The model architecture below MUST match the base checkpoint. SFT loads its
weights with strict=True, so a mismatch fails loudly and immediately.
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-350m-sft-alpaca",

    # ---- model: must match the 350M base checkpoint ----
    n_layer=16,
    n_head=20,
    n_embd=1280,
    block_size=512,         # the 350M base was trained at 512

    # ---- start from the pretrained 350M base ----
    init_from="checkpoints/350m/ckpt_final.pt",

    # ---- SFT data (already prepared by scripts/prepare_sft_data.py) ----
    data_dir="data/alpaca_sft",

    # ---- optimization: same shape as the 50M / 150M SFT, just rescaled LR ----
    # ~51k examples / batch 64 ≈ 800 iters/epoch; 5k iters ≈ 6 epochs.
    batch_size=64,
    grad_accum_steps=1,
    max_iters=5_000,
    lr=6e-5,                # 150M-SFT's 7e-5 × sqrt(1024/1280) ≈ 6.3e-5; rounded down
    min_lr=1e-5,
    schedule="cosine",      # smooth decay, standard for fine-tuning
    warmup_iters=100,

    # ---- evaluation / sampling ----
    eval_interval=200,
    eval_iters=100,         # less noisy val curve than the default 50
    sample_interval=500,

    # ---- system ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/350m_sft_alpaca",

    # ---- logging ----
    # Same wandb_log=True default we now use across pretrain + SFT runs.
    wandb_log=True,
)
