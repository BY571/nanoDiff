#!/usr/bin/env bash
# Run the DGX-Spark capacity sweep — 50M, 150M, 350M on the same 2B FineWeb-Edu
# tokens. ~10-12 hours total on a Spark. Three checkpoints + three wandb runs.
#
# Prereqs:
#   - Data prepared:  python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 2_000_000_000
#   - wandb logged in: wandb login (or WANDB_API_KEY set)
#
# Run from the repo root:
#   bash scripts/spark_sweep.sh

set -euo pipefail

export NANODIFF_WANDB=1
# Reduces PyTorch CUDA-allocator fragmentation. The diffusion loss's
# float-logits tensor is ~26 GB at B=128, V=50304 — without expandable
# segments the allocator leaves ~25 GB reserved-but-unallocated and OOMs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/bin/python"

echo "=== capacity sweep: 50M → 150M → 350M on 2B FineWeb-Edu ==="
date

echo
echo "=== Run 1/3: 50M ==="
$PY -u train.py --config configs/train_50m_spark.py

echo
echo "=== Run 2/3: 150M ==="
$PY -u train.py --config configs/train_150m_spark.py

echo
echo "=== Run 3/3: 350M ==="
$PY -u train.py --config configs/train_350m_spark.py

echo
echo "=== sweep complete ==="
date
ls -lh checkpoints/50m_spark/ checkpoints/150m_spark/ checkpoints/350m_spark/
