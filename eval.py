"""Estimate the NLL upper bound / perplexity of a checkpoint on held-out data.

The diffusion training loss IS a stochastic upper bound on the negative
log-likelihood (LLaDA Eq. 13). Here we Monte-Carlo average it over many random
(t, mask) draws to get a stable estimate, then report perplexity = exp(loss).

    python eval.py --ckpt checkpoints/150m/ckpt.pt --iters 500

Note: LLaDA's Eq. 14 gives a lower-variance estimator (mask exactly `l` tokens for
l ~ U{1..T} instead of i.i.d. masking). We use the simple Eq. 12 form here because
it exactly matches the training objective; implementing Eq. 14 is a good exercise.
"""
import argparse
import math
import os

import torch

from nanodiff.model import NanoDiff
from nanodiff.diffusion import forward_process, diffusion_loss
from nanodiff.data import TokenDataset
from nanodiff.utils import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--iters", type=int, default=500, help="number of MC batches")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt = load_checkpoint(args.ckpt, map_location=args.device)
    cfg = ckpt["config"]
    model = NanoDiff(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    batch_size = args.batch_size or cfg.batch_size
    data = TokenDataset(os.path.join(cfg.data_dir, f"{args.split}.bin"), cfg.block_size)

    losses = torch.zeros(args.iters)
    with torch.no_grad():
        for k in range(args.iters):
            x0 = data.get_batch(batch_size, args.device)
            x_t, mask, t = forward_process(x0, cfg.mask_token_id, cfg.t_eps)
            logits = model(x_t)
            losses[k] = diffusion_loss(logits, x0, mask, t).item()

    loss = losses.mean().item()
    print(f"{args.split}: NLL-bound {loss:.4f} nats/token  |  "
          f"perplexity {math.exp(loss):.2f}  "
          f"(averaged over {args.iters} batches of {batch_size})")


if __name__ == "__main__":
    main()
