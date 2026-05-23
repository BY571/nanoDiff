"""nanoDiff · 350M · capacity-scaling run (10B tokens).

The next rung on the ladder after the 50M/150M scaling comparison.

At matched 3B tokens, the 150M beat the 50M by 0.13 nats (val 3.78 vs 3.91).
This run pushes capacity further: a ~350M-class model on a 10B-token shard.
Chinchilla-optimal for 350M is ~7B tokens; the extra ~40% buys us a single
data shard the next round of scaling experiments can re-use without
re-prepping (and gives the 350M a chance to fully tail off rather than
ending mid-decay).

Architecture choice — wider-shorter (16L × 1280d × 20h, head_dim=64).
Vs a depth-heavy alternative (24L × 1024d × 16h) of the same parameter
count: width is cheaper per-step on a single GPU (better matmul shapes)
and validation loss tends to be flatter to width than to depth at small
scales — so we burn fewer wall-clock hours per gradient update for the
same generalization headroom.

From the repo root:
    python scripts/prepare_data.py --out-dir data/fineweb_edu_10b \\
        --num-tokens 10_000_000_000
    python pretrain/train.py --config pretrain/configs/350m.py
"""
from nanodiff.config import Config

config = Config(
    name="nanodiff-350m",

    # ---- model (~382M total, ~317M non-embedding) ----
    # 16 × 1280 × 20 heads (head_dim = 1280/20 = 64, the canonical "wide" head size)
    n_layer=16,
    n_head=20,
    n_embd=1280,
    block_size=512,

    # ---- data ----
    # The 10B-token FineWeb-Edu shard prepared by the non-streaming prep.
    # The 50M (2B) and 150M (3B) runs use the same shard, so eval val.bin is shared.
    data_dir="data/fineweb_edu_10b",

    # ---- optimization ----
    # Effective batch 256 sequences = 131,072 tokens / step.
    # 10B tokens / 131,072 tok/step = 76,294 steps → round to 76,300.
    # `batch_size` × `grad_accum_steps` splits the effective batch for VRAM headroom;
    # total throughput is invariant under that split. Start here and tighten the
    # microbatch up if the Spark has headroom after the first few hundred iters.
    batch_size=64,
    grad_accum_steps=4,
    max_iters=76_300,
    lr=6e-4,             # ≈ 150M's 8e-4 × √(1024/1280); conservatively rounded down
    min_lr=1e-5,
    warmup_iters=1_000,
    decay_iters=25_000,  # final ~33% of training, matching the 150M's decay fraction

    # ---- evaluation ----
    eval_interval=2_000,  # ~38 evals across the run; cheap relative to the step cost

    # ---- system ----
    device="cuda",
    dtype="bfloat16",
    compile=True,

    # ---- io ----
    out_dir="checkpoints/350m",
)
