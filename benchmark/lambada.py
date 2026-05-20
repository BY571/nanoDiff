"""LAMBADA benchmark for a nanoDiff masked diffusion LM.

LAMBADA (Paperno et al. 2016) tests long-range discourse understanding: each
example is a passage whose final word is easy to guess from the *broad*
context but not from the last sentence alone. The task is to predict that
final word.

Why LAMBADA for small models: multiple-choice benchmarks (HellaSwag, MMLU,
ARC) sit at random chance until models are far larger than nanoDiff's. LAMBADA
last-word prediction has genuine signal at 50-150M params, so it tracks real
progress while the models are small. Once accuracy here saturates, that is the
signal to graduate to the harder standard benchmarks.

Scoring a *diffusion* LM: there is no autoregressive chain rule. We lay down
[context | last-word], mask the last-word tokens, run one forward pass, and
read the predictions at the masked positions:
  * accuracy   — argmax at every last-word position matches the target word
  * perplexity — exp(mean cross-entropy over the target tokens)

This is the single-pass variant (the whole last word masked at once); it is
deterministic and consistent for comparing checkpoints.

    python benchmark/lambada.py --ckpt checkpoints/50m/ckpt.pt
    python benchmark/lambada.py --ckpt <ckpt> --limit 200    # quick subset
"""
import argparse
import math

import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset

from nanodiff.model import NanoDiff
from nanodiff.utils import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N examples (for quick runs)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = load_checkpoint(args.ckpt, map_location=args.device)
    cfg = ckpt["config"]
    model = NanoDiff(cfg)
    model.load_state_dict(ckpt["model"])
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model.to(args.device, dtype=dtype).eval()
    mask_id = cfg.mask_token_id

    enc = tiktoken.get_encoding("gpt2")
    data = load_dataset("EleutherAI/lambada_openai", split="test")
    n = len(data) if args.limit is None else min(args.limit, len(data))
    print(f"LAMBADA · {cfg.name} · {n} examples · device={args.device}\n")

    correct = 0
    evaluated = 0
    total_ce = 0.0
    total_target_tokens = 0

    with torch.no_grad():
        for i in range(n):
            text = data[i]["text"].strip()
            ctx, _, last = text.rpartition(" ")
            if not ctx or not last:
                continue                          # no clear last word — skip
            ctx_ids = enc.encode(ctx)
            target_ids = enc.encode(" " + last)   # leading space: BPE word boundary
            L = len(target_ids)

            # keep [context | target] within the context window; trim context left
            if len(ctx_ids) + L > cfg.block_size:
                ctx_ids = ctx_ids[-(cfg.block_size - L):]
            P = len(ctx_ids)

            x = torch.tensor([ctx_ids + target_ids], dtype=torch.long, device=args.device)
            x[:, P:] = mask_id                    # mask the whole last-word span
            logits = model(x)[0, P:].float()      # (L, V) — predictions for the word
            tgt = torch.tensor(target_ids, device=args.device)

            if (logits.argmax(-1) == tgt).all():
                correct += 1
            total_ce += F.cross_entropy(logits, tgt, reduction="sum").item()
            total_target_tokens += L
            evaluated += 1
            if evaluated % 500 == 0:
                print(f"  {evaluated}/{n}  running acc {correct / evaluated * 100:.1f}%")

    acc = correct / evaluated * 100
    ppl = math.exp(total_ce / total_target_tokens)
    print(f"\nLAMBADA accuracy            : {acc:.2f}%  ({correct}/{evaluated})")
    print(f"LAMBADA perplexity (last word): {ppl:.1f}")


if __name__ == "__main__":
    main()
