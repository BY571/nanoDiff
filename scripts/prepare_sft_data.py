"""Download and tokenize an instruction dataset for supervised fine-tuning.

Default dataset: yahma/alpaca-cleaned (~52k single-turn instruction examples,
the community-curated Alpaca with the original's factual/format errors fixed).

Each example -> a fixed-length (prompt, response) token-id pair via
nanodiff.sft.encode_sft_example. Output: <out-dir>/{train,val}.npz, each a
numpy archive with two uint16 arrays: `prompts` (N, P) and `responses` (N, L).

    python scripts/prepare_sft_data.py --out-dir data/alpaca_sft

The [MASK] token is never stored — the SFT forward process adds it at training
time. The padding / answer-end token is <|endoftext|> (50256).
"""
import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from nanodiff.sft import encode_sft_example

EOT = 50256  # <|endoftext|> — padding + answer end-marker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/alpaca_sft")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned")
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--response-len", type=int, default=256)
    p.add_argument("--val-size", type=int, default=1000,
                   help="examples held out for validation")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    ds = load_dataset(args.dataset, split="train")
    print(f"loaded {len(ds):,} examples from {args.dataset}")

    prompts, responses = [], []
    for ex in tqdm(ds, desc="encoding"):
        p_ids, r_ids = encode_sft_example(
            ex.get("instruction", "") or "",
            ex.get("input", "") or "",
            ex.get("output", "") or "",
            enc, args.prompt_len, args.response_len, EOT)
        prompts.append(p_ids)
        responses.append(r_ids)

    prompts = np.array(prompts, dtype=np.uint16)        # (N, P)
    responses = np.array(responses, dtype=np.uint16)    # (N, L)

    # shuffle once, then split off a fixed validation set
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(prompts))
    prompts, responses = prompts[perm], responses[perm]

    v = min(args.val_size, len(prompts) // 10)
    np.savez(os.path.join(args.out_dir, "val.npz"),
             prompts=prompts[:v], responses=responses[:v])
    np.savez(os.path.join(args.out_dir, "train.npz"),
             prompts=prompts[v:], responses=responses[v:])

    print(f"wrote {len(prompts) - v:,} train / {v:,} val examples -> {args.out_dir}/")
    print(f"  prompt_len={args.prompt_len}  response_len={args.response_len}  "
          f"(sequence length P+L = {args.prompt_len + args.response_len})")


if __name__ == "__main__":
    main()
