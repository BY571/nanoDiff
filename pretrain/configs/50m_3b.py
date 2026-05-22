"""nanoDiff · 50M · matched-token control run (3B tokens).

The 150M run beat the 50M base — but on a confounded comparison: the 150M saw
3B tokens, the 50M base only ~2B. This run trains the *50M* on the same 3B
budget (same data shard, same block_size, same schedule as 150m.py) so the
50M-vs-150M comparison becomes token-matched and the remaining gap can be
attributed to capacity rather than to token count.

Identical to pretrain/configs/150m.py except the model is 50M and the LR is the
50M's established 1.2e-3.

From the repo root:
    python pretrain/train.py --config pretrain/configs/50m_3b.py
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-50m-3b",

    # ---- model (~50M non-embedding params) ----
    n_layer=7,
    n_head=12,
    n_embd=768,
    block_size=512,

    # ---- data (same shard as the 150M run) ----
    data_dir="data/fineweb_edu_10b",

    # ---- optimization (matches 150m.py; only the LR differs) ----
    batch_size=128,
    grad_accum_steps=1,
    max_iters=46_000,       # 46k x 65.5K tokens = ~3.0B, matching the 150M run
    lr=1.2e-3,              # the 50M's established LR
    min_lr=1e-5,
    warmup_iters=1_000,
    decay_iters=15_000,

    # ---- evaluation ----
    eval_interval=1_000,

    # ---- system ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/50m_3b",
)
