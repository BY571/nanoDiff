"""30M model on a 2B-token FineWeb-Edu subset — the "saturate the data axis" run.

At 16K tokens/iter, 125k iters = exactly 1 epoch over 2B tokens (≈ 2.0 G of
.bin on disk). On a Jetson Orin this is ~40-45 hours; on a real GPU box,
~minutes-to-an-hour. Designed to test whether the 30M model is *data-bound*
or has hit its representational floor.

    python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 2_000_000_000
    NANODIFF_WANDB=1 python train.py --config configs/train_30m_2b.py
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-30m-2b",

    # ---- model  (same 30M as configs/train_30m.py) ----
    n_layer=6,
    n_head=6,
    n_embd=384,
    block_size=512,

    # ---- data ----
    data_dir="data/fineweb_edu",

    # ---- optimization ----
    # effective batch = 8 * 4 = 32 sequences (~16K tokens / iter)
    # 125k iters * 16K tokens = 2B tokens = exactly 1 epoch
    batch_size=8,
    grad_accum_steps=4,
    max_iters=125_000,
    lr=6e-4,
    min_lr=1e-5,
    warmup_iters=4_000,        # ~3% — same proportion as the 30k-iter run
    decay_iters=20_000,        # ~16% — same proportion as the 30k-iter run

    # ---- evaluation cadence ----
    # eval costs ~50s on the Jetson; at 125k iters we want fewer evals to keep
    # overhead reasonable (~50 evals * ~50s = ~40 min total eval, ~1.5% overhead).
    eval_interval=2_500,

    # ---- io ----
    out_dir="checkpoints/30m_2b",
)
