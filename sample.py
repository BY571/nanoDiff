"""Sample text from a trained nanoDiff checkpoint.

    # pure diffusion, greedy
    python sample.py --ckpt checkpoints/150m/ckpt.pt --prompt "The meaning of life is"

    # semi-autoregressive (block) decoding, with sampling
    python sample.py --ckpt checkpoints/150m/ckpt.pt --prompt "Once upon a time" \\
        --gen-length 256 --steps 256 --block-length 32 --temperature 0.7
"""
import argparse

import torch
import tiktoken

from nanodiff.model import NanoDiff
from nanodiff.sampler import generate
from nanodiff.utils import load_checkpoint

EOT = 50256  # <|endoftext|> — ids below this are ordinary text; at/above are special


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="", help="conditioning text (empty -> unconditional)")
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--steps", type=int, default=128, help="total denoising steps")
    p.add_argument("--block-length", type=int, default=None,
                   help="semi-AR block size; default = gen-length (pure diffusion)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy argmax; >0 = sample")
    p.add_argument("--remasking", default="low_confidence",
                   choices=["low_confidence", "random"])
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt = load_checkpoint(args.ckpt, map_location=args.device)
    cfg = ckpt["config"]
    model = NanoDiff(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    enc = tiktoken.get_encoding("gpt2")
    if args.prompt:
        prompt_ids = enc.encode(args.prompt, allowed_special={"<|endoftext|>"})
    else:
        prompt_ids = [EOT]                       # unconditional: start from <|endoftext|>
    prompt = torch.tensor([prompt_ids] * args.num_samples,
                          dtype=torch.long, device=args.device)

    out = generate(model, prompt, args.gen_length, steps=args.steps,
                   block_length=args.block_length, temperature=args.temperature,
                   remasking=args.remasking)

    for i in range(args.num_samples):
        ids = [tok for tok in out[i].tolist() if tok < EOT]   # drop special/mask tokens
        print(f"\n--- sample {i + 1}/{args.num_samples} ---")
        print(enc.decode(ids))


if __name__ == "__main__":
    main()
