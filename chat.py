"""Interactive REPL for a nanoDiff model.

    python chat.py --ckpt checkpoints/base/ckpt.pt          # base model
    python chat.py --ckpt checkpoints/sft/ckpt.pt --sft     # SFT'd model

Two modes:
  * default — base diffusion LM. A "conversation" is one continuous document
    that grows turn by turn: your input is appended, the model continues it,
    the continuation is appended back. Prompt it document-style.
  * --sft — instruction-tuned model. Each turn is an independent instruction,
    wrapped in the SFT prompt template (### Instruction / ### Response); no
    history is accumulated, and the answer is cut at the model's EOT marker.

Tip: crank temperature for small/under-trained models that loop on repetition.

REPL commands:
    >>> <text>                    a turn — a doc fragment (base) or instruction (sft)
    >>> !set <key> <value>        change a knob — gen-length, steps, block-length,
                                  temperature, top-p, rep-penalty, seed
    >>> !reset                    clear conversation history (base mode)
    >>> !history                  print the running document (base mode)
    >>> q / quit / exit           leave
"""
import argparse
import time

import tiktoken
import torch

from nanodiff.model import NanoDiff
from nanodiff.sampler import generate
from nanodiff.sft import SFT_PROMPT_NO_INPUT
from nanodiff.utils import load_checkpoint

EOT = 50256  # <|endoftext|>; real text tokens are < EOT (50257=MASK, 50258+=padding)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sft", action="store_true",
                   help="instruction-tuned model: wrap each turn in the SFT prompt "
                        "template and treat it as a fresh, independent instruction.")
    p.add_argument("--gen-length", type=int, default=96)
    p.add_argument("--steps", type=int, default=32,
                   help="denoising steps. Default 32 (3 commits/step at gen=96) — the "
                        "validated fast preset with the sampler's within-step rep_penalty "
                        "preventing word-doubling. Bump to 96 (==gen-length) for the "
                        "LLaDA-recommended one-commit-per-step quality maximum (~3x slower).")
    p.add_argument("--block-length", type=int, default=32,
                   help="semi-AR block size (<= gen-length); smaller = more AR-like. "
                        "Default 32 pairs with steps=32 (3 commits/step, 3 blocks of "
                        "32). Smaller values like 16 force the model to wrap responses "
                        "inside the first block — visible as short answers on casual "
                        "prompts. Larger values trade per-block parallelism but give "
                        "the model more 'lookahead' for EOT placement.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None,
                   help="keep only the k highest-prob tokens; off by default")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="nucleus sampling cutoff; trims the unlikely tail. 1.0 disables.")
    p.add_argument("--rep-penalty", type=float, default=3.0,
                   help="logit penalty on already-present tokens; the real cure "
                        "for repetition collapse on small diffusion LMs. 0 disables.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default=None,
                   help="bfloat16/float16/float32; auto = bf16 on cuda, fp32 on cpu. "
                        "Matches training dtype for ~2x speedup on GPU.")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile(model) for kernel fusion. Default ON — ~1.4x "
                        "faster, stacks with --steps 32 to ~4.4x. One-time ~5-30s "
                        "warmup on the first generation (depends on Inductor cache "
                        "state). Use --no-compile to disable (e.g. for a one-off "
                        "query where the warmup isn't worth it, or on a build without "
                        "Triton). Skip --use-cache when using --compile — they target "
                        "the same overhead and don't stack.")
    p.add_argument("--use-cache", action="store_true",
                   help="Fast-dLLM prefix K/V cache. Approximate (LAMBADA -0.02pp); "
                        "helps most at gen>=256 or batch>=4. Off by default.")
    p.add_argument("--tau", type=float, default=None,
                   help="Fast-dLLM threshold parallel decoding (Lou et al. 2025). "
                        "Commit every active position with confidence >= tau instead "
                        "of the fixed schedule. Best with --use-cache. tau=0.5 gives "
                        "~1.45x speedup on the default workload; lower = faster, "
                        "looser quality bar.")
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
        # torch.compile is default-on for the fast preset, but the surrounding
        # ergonomics matter: any environment without a working Triton (some
        # Jetson / ARM builds, very old PyTorch) would otherwise crash before
        # the user sees a prompt. Suppress dynamo errors so we degrade to
        # eager instead of dying. The first generation prints the warmup
        # delay; subsequent turns are fast.
        #
        # Also silence the cosmetic "Not enough SMs to use max_autotune_gemm
        # mode" warning that inductor prints on sm_121 (Blackwell GB10). We
        # use mode="default", not max_autotune, so the gate is irrelevant —
        # but the warning fires every torch.compile() call, polluting the
        # chat startup output. A targeted message filter leaves other inductor
        # diagnostics visible. (See IMPROVEMENTS.md for the full sm_121 story.)
        import logging
        logging.getLogger("torch._inductor.utils").addFilter(
            lambda r: "Not enough SMs" not in r.getMessage()
        )
        print("compiling kernels (one-time ~5-30s warmup on first generation; "
              "pass --no-compile to skip)…")
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            model = torch.compile(model)
        except Exception as e:
            print(f"  torch.compile unavailable, falling back to eager: {e}")

    enc = tiktoken.get_encoding("gpt2")

    n_params = (sum(p.numel() for p in model.parameters())) / 1e6
    print(f"\nnanoDiff chat  ·  {cfg.name}  ·  {n_params:.0f}M params  ·  "
          f"device={args.device}  dtype={args.dtype}  compile={args.compile}")
    print(f"gen_length={args.gen_length}  steps={args.steps}  "
          f"block_length={args.block_length}  temperature={args.temperature}  "
          f"top_k={args.top_k}  top_p={args.top_p}  rep_penalty={args.rep_penalty}")
    if args.sft:
        print("SFT model — each turn is an independent instruction (no history).")
    else:
        print("BASE model (no instruction tuning) — turns accumulate as one document.")
    print("Commands: `!reset` to clear, `!history`, `!set <key> <value>`, `q` to quit.\n")

    settable = {"gen_length", "steps", "block_length", "temperature",
                "top_k", "top_p", "rep_penalty", "seed"}
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

        # Build this turn's prompt.
        if args.sft:
            # SFT model: wrap the instruction in the template; no history.
            prompt_str = SFT_PROMPT_NO_INPUT.format(instruction=user)
            prompt_ids = enc.encode(prompt_str, allowed_special={"<|endoftext|>"})
        else:
            # Base model: append the turn to the running document (newline-sep).
            new_user_ids = enc.encode(user, allowed_special={"<|endoftext|>"})
            sep = enc.encode("\n") if history_ids else []
            history_ids = history_ids + sep + new_user_ids
            # Keep the prompt within the context window; slide from the left.
            max_prompt_len = cfg.block_size - args.gen_length
            if max_prompt_len <= 0:
                raise ValueError("gen_length must be < block_size")
            if len(history_ids) > max_prompt_len:
                history_ids = history_ids[-max_prompt_len:]
            prompt_ids = history_ids

        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        out = generate(
            model, prompt,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
            use_cache=args.use_cache,
            tau=args.tau,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0

        gen = out[0, len(prompt_ids):].tolist()
        # SFT: the model is trained to emit <|endoftext|> at the answer's end —
        # truncate there to cut the junk tail a weak model rambles into.
        if args.sft and EOT in gen:
            gen = gen[:gen.index(EOT)]
        gen_ids = [t for t in gen if t < EOT]   # drop MASK / padding / stray specials
        print(f"<<< {enc.decode(gen_ids)}\n")
        # Timing on its own dimmed line — ANSI \033[2m = dim, \033[0m = reset —
        # blank line above and below so it reads as metadata, not the answer.
        # tok/s is RAW (gen_length / dt), not post-truncation: the model did
        # `gen_length` tokens of work regardless of how many survive EOT
        # truncation. The "answer N tok" suffix shows the post-trim count.
        print(f"\033[2m({dt * 1000:.0f} ms · {args.gen_length / dt:.0f} tok/s"
              f" · answer {len(gen_ids)} tok)\033[0m\n")

        # Base model: append the continuation so the next turn sees it.
        if not args.sft:
            history_ids = history_ids + gen_ids


if __name__ == "__main__":
    main()
