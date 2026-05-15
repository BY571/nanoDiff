"""Interactive REPL — type a prompt, the model continues it.

    python chat.py --ckpt checkpoints/30m/ckpt.pt

Important: nanoDiff is a *base* diffusion language model — it does next-text
completion, not instruction following. Typing "what is python?" will not yield
an answer; it will continue the surface form, the way GPT-2 base does. Real
chat behavior requires SFT (see nanodiff/sft.py).

Tip: try prompts like a sentence start ("The history of diffusion models began"),
or paste a paragraph and let the model finish the next sentences.

REPL commands:
    >>> <text>                    generate a continuation
    >>> !set <key> <value>        change a knob, e.g. `!set temperature 1.0`
    >>> q  /  quit  /  exit       leave
Knobs you can !set: gen-length, steps, block-length, temperature, seed
"""
import argparse

import tiktoken
import torch

from nanodiff.model import NanoDiff
from nanodiff.sampler import generate
from nanodiff.utils import load_checkpoint

EOT = 50256  # <|endoftext|>; real text tokens are all < EOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--steps", type=int, default=256,
                   help="denoising steps; >= gen-length for good quality")
    p.add_argument("--block-length", type=int, default=32,
                   help="semi-AR block size (<= gen-length); smaller = more AR-like")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = load_checkpoint(args.ckpt, map_location=args.device)
    cfg = ckpt["config"]
    model = NanoDiff(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()
    enc = tiktoken.get_encoding("gpt2")

    n_params = model.get_num_params(non_embedding=False) / 1e6
    print(f"\nnanoDiff chat  ·  {cfg.name}  ·  {n_params:.0f}M params  ·  device={args.device}")
    print(f"gen_length={args.gen_length}  steps={args.steps}  "
          f"block_length={args.block_length}  temperature={args.temperature}")
    print("BASE model (no instruction tuning) — type a prompt to continue it.")
    print("Commands: `!set <key> <value>` to tune, `q` to quit.\n")

    settable = {"gen_length", "steps", "block_length", "temperature", "seed"}

    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"q", "quit", "exit"}:
            break
        if user.startswith("!set "):
            try:
                _, key, value = user.split(maxsplit=2)
                attr = key.replace("-", "_")
                if attr not in settable:
                    print(f"  unknown knob `{key}`; try one of {sorted(settable)}")
                    continue
                cur = getattr(args, attr)
                setattr(args, attr, type(cur)(value) if cur is not None else int(value))
                print(f"  {attr} = {getattr(args, attr)}")
            except ValueError as e:
                print(f"  parse error: {e}")
            continue

        prompt_ids = enc.encode(user, allowed_special={"<|endoftext|>"})
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        out = generate(
            model, prompt,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
        )
        gen_ids = [t for t in out[0, len(prompt_ids):].tolist() if t < EOT]
        print(user + enc.decode(gen_ids) + "\n")


if __name__ == "__main__":
    main()
