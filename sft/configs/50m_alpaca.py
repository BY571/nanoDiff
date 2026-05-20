"""nanoDiff · 50M · SFT config — instruction-tune the base on Alpaca-cleaned.

Fine-tunes the 50M base checkpoint into an instruction-follower with the LLaDA
Algorithm 2 recipe (response-only masking — see nanodiff/sft.py). This config
trained the public SFT checkpoint
[Sebasdi/nanodiff-50m-sft-alpaca](https://huggingface.co/Sebasdi/nanodiff-50m-sft-alpaca).

From the repo root:
    python scripts/prepare_sft_data.py --out-dir data/alpaca_sft
    python sft/train.py --config sft/configs/50m_alpaca.py

The model architecture below MUST match the base checkpoint — SFT loads its
weights with strict=True, so a mismatch fails loudly and immediately.
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
    init_from="checkpoints/50m/ckpt_final.pt",

    # ---- SFT data (from scripts/prepare_sft_data.py) ----
    data_dir="data/alpaca_sft",

    # ---- optimization — fine-tuning, not pretraining: low LR, cosine decay ----
    # ~51k examples / batch 64 is ~800 iters/epoch. The instruction *format* is
    # learned within ~200 iters, but the loss keeps a slow, real descent well
    # past that — a 2.4k-iter run had not saturated, so we give it 5k.
    batch_size=64,
    grad_accum_steps=1,
    max_iters=5_000,
    lr=1e-4,                # ~10x below the base run's 1.2e-3
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
    out_dir="checkpoints/50m_sft_alpaca",
)
