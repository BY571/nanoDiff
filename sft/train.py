"""Supervised fine-tuning of a nanoDiff base model on instruction data.

    python train_sft.py --config configs/sft_50m_spark.py

SFT is pretraining with two changes (LLaDA Algorithm 2):
  * the forward process masks only the RESPONSE span — the prompt is clean
    conditioning (see nanodiff/sft.py);
  * training starts from a pretrained base checkpoint (`cfg.init_from`), with
    a fresh optimizer, rather than from random init.

Single GPU only — SFT is short (a few thousand iters), so no DDP scaffolding.
"""
import argparse
import os
import time
from contextlib import nullcontext

import torch

from nanodiff.model import NanoDiff
from nanodiff.sft import sft_forward_process, sft_loss, SFT_PROMPT_NO_INPUT
from nanodiff.data import SFTDataset
from nanodiff.sampler import generate
from nanodiff.utils import (load_config, load_checkpoint, get_lr,
                            save_checkpoint, unwrap_model, human)

EOT = 50256  # <|endoftext|>


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a configs/*.py file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = cfg.device
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[cfg.dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    # ---- data ----
    train_data = SFTDataset(os.path.join(cfg.data_dir, "train.npz"))
    val_data = SFTDataset(os.path.join(cfg.data_dir, "val.npz"))
    print(f"config        : {args.config}")
    print(f"SFT examples  : {len(train_data):,} train / {len(val_data):,} val")

    # ---- model: build, then load the pretrained base weights ----
    if not cfg.init_from:
        raise ValueError("SFT needs cfg.init_from — the base checkpoint to fine-tune")
    model = NanoDiff(cfg).to(device)
    base = load_checkpoint(cfg.init_from, map_location=device)
    model.load_state_dict(base["model"])      # strict: arch must match the base
    print(f"init_from     : {cfg.init_from}  (base best_val {base.get('best_val')})")
    print(f"params        : {human(model.get_num_params(False))} total")

    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)

    if cfg.compile:
        print(f"compiling model (mode={cfg.compile_mode!r}; first step slow) ...")
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        model = torch.compile(model, mode=cfg.compile_mode)
    raw_model = unwrap_model(model)
    scaler = torch.amp.GradScaler(device_type, enabled=(cfg.dtype == "float16"))

    use_wandb = (
        cfg.wandb_log
        or os.environ.get("NANODIFF_WANDB", "").lower() in ("1", "true", "yes", "on")
    )
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.name, config=vars(cfg))

    @torch.no_grad()
    def estimate_loss():
        """Average SFT loss on the train/val instruction splits."""
        model.eval()
        out = {}
        for split, dataset in [("train", train_data), ("val", val_data)]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                p, r = dataset.get_batch(cfg.batch_size, device)
                x_t, resp_mask, t = sft_forward_process(p, r, cfg.mask_token_id, cfg.t_eps)
                with ctx:
                    logits = model(x_t)
                    losses[k] = sft_loss(logits, r, resp_mask, t).item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def sanity_sample():
        """Generate an answer to a fixed instruction so you can eyeball SFT progress."""
        try:
            import tiktoken
        except ImportError:
            return
        enc = tiktoken.get_encoding("gpt2")
        instr = "Name three primary colors."
        prompt_str = SFT_PROMPT_NO_INPUT.format(instruction=instr)
        pids = torch.tensor([enc.encode(prompt_str)], dtype=torch.long, device=device)
        with ctx:
            out = generate(raw_model, pids, gen_length=64, steps=64,
                           block_length=32, temperature=0.0, rep_penalty=3.0)
        gen = [tok for tok in out[0, pids.shape[1]:].tolist() if tok < EOT]
        text = enc.decode(gen)
        print(f"  [{instr}] -> {text!r}")
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

        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss()
            print(f"iter {iter_num:>6}: train {losses['train']:.4f}  "
                  f"val {losses['val']:.4f}  lr {lr:.2e}")
            if use_wandb:
                wandb.log({"train/loss": losses["train"], "val/loss": losses["val"],
                           "lr": lr}, step=iter_num)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(os.path.join(cfg.out_dir, "ckpt.pt"),
                                model, optimizer, iter_num, cfg, best_val)

        if iter_num % cfg.sample_interval == 0 and iter_num > 0:
            sanity_sample()

        # forward / backward with gradient accumulation
        for _ in range(cfg.grad_accum_steps):
            p, r = train_data.get_batch(cfg.batch_size, device)
            x_t, resp_mask, t = sft_forward_process(p, r, cfg.mask_token_id, cfg.t_eps)
            with ctx:
                logits = model(x_t)
                loss = sft_loss(logits, r, resp_mask, t) / cfg.grad_accum_steps
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1

        if iter_num % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            full_loss = loss.item() * cfg.grad_accum_steps
            ms_per_step = dt / cfg.log_interval * 1000
            print(f"  step {iter_num:>6}  loss {full_loss:.4f}  {ms_per_step:.0f} ms/step")
            if use_wandb:
                wandb.log({"train/step_loss": full_loss, "lr": lr,
                           "perf/ms_per_step": ms_per_step}, step=iter_num)

    save_checkpoint(os.path.join(cfg.out_dir, "ckpt_final.pt"),
                    model, optimizer, iter_num, cfg, best_val)
    print(f"done. best val loss {best_val:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
