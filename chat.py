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
    p.add_argument("--steps", type=int, default=96,
                   help="denoising steps; LLaDA-recommended sweet spot is steps≈gen-length. "
                        "Set higher for quality (slower), lower for speed (quality drops fast).")
    p.add_argument("--block-length", type=int, default=32,
                   help="semi-AR block size (<= gen-length); smaller = more AR-like")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None,
                   help="keep only the k highest-prob tokens; off by default")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="nucleus sampling cutoff; the main lever against "
                        "repetition collapse on small base models. 1.0 disables.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default=None,
                   help="bfloat16/float16/float32; auto = bf16 on cuda, fp32 on cpu. "
                        "Matches training dtype for ~2x speedup on GPU.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(model) for kernel fusion. ~2-3x faster after a "
                        "one-time ~20s warmup on the first generation.")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = load_checkpoint(args.ckpt, map_location=args.device)
    cfg = ckpt["config"]
    model = NanoDiff(cfg)
    model.load_state_dict(ckpt["model"])

    # Match training dtype on GPU; fp32 weights are 2x larger and 2x slower on Blackwell.
    if args.dtype is None:
        args.dtype = "bfloat16" if args.device.startswith("cuda") else "float32"
    dtype = getattr(torch, args.dtype)
    model.to(args.device, dtype=dtype).eval()

    if args.compile:
        print("compiling kernels (one-time ~20s warmup on first generation)…")
        model = torch.compile(model)

    enc = tiktoken.get_encoding("gpt2")

    n_params = (sum(p.numel() for p in model.parameters())) / 1e6
    print(f"\nnanoDiff chat  ·  {cfg.name}  ·  {n_params:.0f}M params  ·  "
          f"device={args.device}  dtype={args.dtype}  compile={args.compile}")
    print(f"gen_length={args.gen_length}  steps={args.steps}  "
          f"block_length={args.block_length}  temperature={args.temperature}  "
          f"top_k={args.top_k}  top_p={args.top_p}")
    print("BASE model (no instruction tuning) — your turns accumulate as one document.")
    print("Commands: `!reset` to clear, `!history`, `!set <key> <value>`, `q` to quit.\n")

    settable = {"gen_length", "steps", "block_length", "temperature",
                "top_k", "top_p", "seed"}
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
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        import time as _t; _t0 = _t.time()
        out = generate(
            model, prompt,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = _t.time() - _t0
        # Generated portion only; strip special tokens (MASK/padding/EOT).
        gen_ids = [t for t in out[0, len(history_ids):].tolist() if t < EOT]
        print(f"<<< [{dt*1000:.0f} ms · {len(gen_ids)/dt:.1f} tok/s] ",
              enc.decode(gen_ids), "\n", sep="")

        # Append the model's continuation to history so the next turn sees it.
        history_ids = history_ids + gen_ids


if __name__ == "__main__":
    main()
