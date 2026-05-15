"""Interactive REPL with conversation memory — each turn sees the running history.

    python chat.py --ckpt checkpoints/30m/ckpt.pt

Mental model: nanoDiff is a *base* diffusion LM (no SFT), so a "conversation" here
is really one continuous document that grows turn by turn. Your input is appended,
the model continues it, and the continuation is appended back. Real chat behavior
(User/Assistant roles, instruction following) needs SFT — see nanodiff/sft.py.

Tip: try long-form prompts that benefit from continuation. Crank temperature for
small/under-trained models that loop on repetition.

REPL commands:
    >>> <text>                    append `<text>` to history, generate continuation
    >>> !set <key> <value>        change a knob — gen-length, steps, block-length,
                                  temperature, seed
    >>> !reset                    clear conversation history
    >>> !history                  print the full running document
    >>> q / quit / exit           leave
"""
import argparse

import tiktoken
import torch

from nanodiff.model import NanoDiff
from nanodiff.sampler import generate
from nanodiff.utils import load_checkpoint

EOT = 50256  # <|endoftext|>; real text tokens are < EOT (50257=MASK, 50258+=padding)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gen-length", type=int, default=96)
    p.add_argument("--steps", type=int, default=192,
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
    print("BASE model (no instruction tuning) — your turns accumulate as one document.")
    print("Commands: `!reset` to clear, `!history`, `!set <key> <value>`, `q` to quit.\n")

    settable = {"gen_length", "steps", "block_length", "temperature", "seed"}
    history_ids: list[int] = []   # accumulated token ids across turns

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
        if user == "!reset":
            history_ids = []
            print("  (history cleared)\n")
            continue
        if user == "!history":
            print("--- running document ---")
            print(enc.decode(history_ids) if history_ids else "(empty)")
            print("------------------------\n")
            continue
        if user.startswith("!set "):
            try:
                _, key, value = user.split(maxsplit=2)
                attr = key.replace("-", "_")
                if attr not in settable:
                    print(f"  unknown knob `{key}`; try one of {sorted(settable)}")
                    continue
                cur = getattr(args, attr)
                setattr(args, attr, type(cur)(value) if cur is not None else int(value))
                print(f"  {attr} = {getattr(args, attr)}\n")
            except ValueError as e:
                print(f"  parse error: {e}\n")
            continue

        # Append the user's turn to the running document (newline-separated).
        new_user_ids = enc.encode(user, allowed_special={"<|endoftext|>"})
        sep = enc.encode("\n") if history_ids else []
        history_ids = history_ids + sep + new_user_ids

        # Keep prompt within the model's context window, leaving room for the
        # generation. Slide the window from the left if needed.
        max_prompt_len = cfg.block_size - args.gen_length
        if max_prompt_len <= 0:
            raise ValueError("gen_length must be < block_size")
        if len(history_ids) > max_prompt_len:
            history_ids = history_ids[-max_prompt_len:]

        prompt = torch.tensor([history_ids], dtype=torch.long, device=args.device)
        out = generate(
            model, prompt,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
        )
        # Generated portion only; strip special tokens (MASK/padding/EOT).
        gen_ids = [t for t in out[0, len(history_ids):].tolist() if t < EOT]
        print("<<<", enc.decode(gen_ids), "\n", sep="")

        # Append the model's continuation to history so the next turn sees it.
        history_ids = history_ids + gen_ids


if __name__ == "__main__":
    main()
