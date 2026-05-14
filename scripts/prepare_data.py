"""Download and tokenize a pretraining corpus into uint16 .bin shards.

Default corpus: a slice of HuggingFaceFW/fineweb-edu (streamed, so no full download).
Tokenizer: the GPT-2 BPE via tiktoken. Output: <out-dir>/train.bin and <out-dir>/val.bin,
each a flat little-endian uint16 array of token ids that nanodiff/data.py memory-maps.

    python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 2_000_000_000

Tokens are valid GPT-2 ids in [0, 50256]; 50256 = <|endoftext|> separates documents.
The [MASK] token (50257) is never stored on disk — it is added by the forward
process at training time.
"""
import argparse
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

EOT = 50256  # <|endoftext|>


def write_split(path, token_iter, target, desc):
    """Stream-encode documents into `path` until ~`target` tokens are written."""
    written = 0
    pbar = tqdm(total=target, desc=desc, unit="tok", unit_scale=True)
    with open(path, "wb") as f:
        for ids in token_iter:
            chunk = np.asarray(ids, dtype=np.uint16)
            f.write(chunk.tobytes())
            written += len(chunk)
            pbar.update(len(chunk))
            if written >= target:
                break
    pbar.close()
    print(f"  wrote {written:,} tokens -> {path}")
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/fineweb_edu")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--num-tokens", type=int, default=2_000_000_000,
                   help="approx number of training tokens to keep")
    p.add_argument("--val-fraction", type=float, default=0.0005)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)

    def token_stream():
        """Yield per-document token id lists, each terminated by <|endoftext|>."""
        for doc in dataset:
            ids = enc.encode_ordinary(doc["text"])
            ids.append(EOT)
            yield ids

    stream = token_stream()
    val_target = max(1, int(args.num_tokens * args.val_fraction))

    # validation split first so it never overlaps with training data
    write_split(os.path.join(args.out_dir, "val.bin"), stream, val_target, "val")
    write_split(os.path.join(args.out_dir, "train.bin"), stream, args.num_tokens, "train")
    print("done.")

    # The `datasets` streaming reader leaves background threads that can crash
    # during interpreter shutdown (a known datasets/fsspec teardown issue). The
    # .bin files are already closed and flushed, so exit hard to skip finalization.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
