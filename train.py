"""Train a nanoDiff masked diffusion language model.

Single GPU:
    python train.py --config configs/train_150m.py

Multi-GPU / DGX-Spark node (torchrun):
    torchrun --standalone --nproc_per_node=8 train.py --config configs/train_1b.py

The training step is the whole point, and it is short:
    x0          = a clean window of tokens
    x_t, mask, t = forward_process(x0)        # corrupt it
    logits      = model(x_t)                  # predict every token
    loss        = diffusion_loss(...)         # 1/t-weighted CE on masked positions
"""
import argparse
import os
import time
from contextlib import nullcontext

import torch

from nanodiff.model import NanoDiff
from nanodiff.diffusion import forward_process, diffusion_loss
from nanodiff.data import TokenDataset
from nanodiff.sampler import generate
from nanodiff.utils import load_config, get_lr, save_checkpoint, unwrap_model, human


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a configs/*.py file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # ---- distributed setup (no-op unless launched with torchrun) ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        # Imported lazily: some PyTorch builds (e.g. Jetson) ship without
        # distributed support — single-GPU training must not require it.
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.distributed import init_process_group, destroy_process_group
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        device = cfg.device
        ddp_world_size = 1
        master_process = True
        seed_offset = 0

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[cfg.dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    tokens_per_iter = (cfg.batch_size * cfg.grad_accum_steps
                       * ddp_world_size * cfg.block_size)
    if master_process:
        print(f"config        : {args.config}")
        print(f"world size    : {ddp_world_size}")
        print(f"tokens / iter : {human(tokens_per_iter)}")

    # ---- data ----
    train_data = TokenDataset(os.path.join(cfg.data_dir, "train.bin"), cfg.block_size)
    val_data = TokenDataset(os.path.join(cfg.data_dir, "val.bin"), cfg.block_size)

    # ---- model + optimizer ----
    model = NanoDiff(cfg).to(device)
    if master_process:
        print(f"params        : {human(model.get_num_params(False))} total, "
              f"{human(model.get_num_params(True))} non-embedding")
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)

    if cfg.compile:
        if master_process:
            print("compiling model (first step will be slow) ...")
        # compile is an optimization, not a requirement — if the backend is
        # unavailable (e.g. a broken/missing triton on some Jetson/ARM builds),
        # fall back to eager instead of crashing the whole run.
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = unwrap_model(model)
    scaler = torch.amp.GradScaler(device_type, enabled=(cfg.dtype == "float16"))

    # ---- logging (Weights & Biases, optional — off unless cfg.wandb_log) ----
    use_wandb = cfg.wandb_log and master_process
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.name, config=vars(cfg))

    @torch.no_grad()
    def estimate_loss():
        """Average diffusion loss on train/val — this IS the NLL upper bound."""
        model.eval()
        out = {}
        for split, dataset in [("train", train_data), ("val", val_data)]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                x0 = dataset.get_batch(cfg.batch_size, device)
                x_t, mask, t = forward_process(x0, cfg.mask_token_id, cfg.t_eps)
                with ctx:
                    logits = model(x_t)
                    losses[k] = diffusion_loss(logits, x0, mask, t).item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def sanity_sample():
        """Generate a short unconditional sample so you can eyeball progress."""
        try:
            import tiktoken
        except ImportError:
            print("  (skipping sample: tiktoken not installed)")
            return
        enc = tiktoken.get_encoding("gpt2")
        prompt = torch.tensor([[50256]], dtype=torch.long, device=device)  # <|endoftext|>
        with ctx:
            out = generate(raw_model, prompt, cfg.sample_gen_length,
                           steps=cfg.sample_steps)
        gen = [tok for tok in out[0].tolist() if tok < 50256]
        text = enc.decode(gen)
        print(f"  sample: {text!r}")
        if use_wandb:
            wandb.log({"sample": text}, step=iter_num)

    # ---- training loop ----
    iter_num = 0
    best_val = float("inf")
    model.train()
    t0 = time.time()

    while iter_num < cfg.max_iters:
        lr = get_lr(iter_num, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # periodic eval + checkpoint
        if iter_num % cfg.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"iter {iter_num:>7}: train {losses['train']:.4f}  "
                  f"val {losses['val']:.4f}  lr {lr:.2e}")
            if use_wandb:
                wandb.log({"train/loss": losses["train"], "val/loss": losses["val"],
                           "lr": lr}, step=iter_num)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(os.path.join(cfg.out_dir, "ckpt.pt"),
                                model, optimizer, iter_num, cfg, best_val)

        # periodic sanity sample
        if iter_num % cfg.sample_interval == 0 and iter_num > 0 and master_process:
            sanity_sample()

        # forward / backward with gradient accumulation
        for micro in range(cfg.grad_accum_steps):
            if ddp:
                # only sync gradients across ranks on the last micro-step
                model.require_backward_grad_sync = (micro == cfg.grad_accum_steps - 1)
            x0 = train_data.get_batch(cfg.batch_size, device)
            x_t, mask, t = forward_process(x0, cfg.mask_token_id, cfg.t_eps)
            with ctx:
                logits = model(x_t)
                loss = diffusion_loss(logits, x0, mask, t) / cfg.grad_accum_steps
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1

        if iter_num % cfg.log_interval == 0 and master_process:
            dt = time.time() - t0
            t0 = time.time()
            full_loss = loss.item() * cfg.grad_accum_steps
            ms_per_step = dt / cfg.log_interval * 1000
            print(f"  step {iter_num:>7}  loss {full_loss:.4f}  {ms_per_step:.0f} ms/step")
            if use_wandb:
                wandb.log({"train/step_loss": full_loss, "lr": lr,
                           "perf/ms_per_step": ms_per_step}, step=iter_num)

    if master_process:
        save_checkpoint(os.path.join(cfg.out_dir, "ckpt_final.pt"),
                        model, optimizer, iter_num, cfg, best_val)
        print(f"done. best val loss {best_val:.4f}")
        if use_wandb:
            wandb.finish()
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
