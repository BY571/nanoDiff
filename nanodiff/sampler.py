"""Reverse process: generating text by iterative denoising.

Implements LLaDA's low-confidence remasking sampler (Algorithm 5), with optional
semi-autoregressive block decoding (LLaDA Fig. 4c).

One denoising step:
  1. Run the model on the current (partly-masked) sequence -> predict every token.
  2. Commit only the most *confident* predictions; leave the rest masked for now.
  3. Repeat. Easy tokens get decided early; hard tokens get more rounds of context
     before they are committed.

`block_length` interpolates between the two regimes (this is the AR<->diffusion
knob that papers like BD3-LM make explicit):
    block_length == gen_length  -> pure diffusion (fill everything in parallel)
    block_length == 1           -> (almost) autoregressive, strictly left to right
"""
import torch
import torch.nn.functional as F


def _transfer_schedule(block_length, steps, device):
    """How many tokens to commit (un-mask) at each denoising step of a block.

    Spreads `block_length` commits as evenly as possible over `steps` steps; the
    counts sum exactly to `block_length`, so a block is fully decoded after its
    last step.
    """
    base = block_length // steps
    rem = block_length % steps
    counts = torch.full((steps,), base, dtype=torch.long, device=device)
    counts[:rem] += 1
    return counts


@torch.no_grad()
def generate(model, prompt_ids, gen_length, steps, block_length=None,
             temperature=0.0, remasking="low_confidence", mask_token_id=None):
    """Generate `gen_length` tokens conditioned on `prompt_ids`.

    Args:
        model:        a NanoDiff (or unwrapped) model
        prompt_ids:   (B, P) LongTensor — the conditioning prefix (never masked)
        gen_length:   number of tokens to generate after the prompt
        steps:        total denoising steps (split evenly across blocks)
        block_length: semi-AR block size; defaults to gen_length (pure diffusion)
        temperature:  0.0 -> greedy argmax; >0 -> sample from the softmax
        remasking:    "low_confidence" (LLaDA default) or "random"
        mask_token_id: override; defaults to model.config.mask_token_id

    Returns:
        (B, P + gen_length) LongTensor — prompt followed by the generated tokens.
    """
    was_training = model.training
    model.eval()
    device = prompt_ids.device
    cfg = model.config
    mask_id = cfg.mask_token_id if mask_token_id is None else mask_token_id
    B, P = prompt_ids.shape

    block_length = block_length or gen_length
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    n_blocks = gen_length // block_length
    assert steps >= n_blocks, "steps must be at least the number of blocks"
    # Distribute `steps` across blocks; if it doesn't divide evenly, the first
    # `extra` blocks get one additional refinement pass each.
    base_steps = steps // n_blocks
    extra = steps % n_blocks

    # Start fully masked, then drop in the (clean, fixed) prompt.
    x = torch.full((B, P + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    for b in range(n_blocks):
        s0 = P + b * block_length          # active block: [s0, s1)
        s1 = s0 + block_length
        block_steps = base_steps + (1 if b < extra else 0)
        counts = _transfer_schedule(block_length, block_steps, device)

        for i in range(block_steps):
            is_mask = x == mask_id
            logits = model(x)
            # The [MASK] token is an input-only sentinel — it never appears in
            # clean data, so it is never a valid generation. Forbid it, otherwise
            # the model could "predict" a mask, the commit becomes a silent no-op,
            # and the position is left masked forever.
            logits[:, :, mask_id] = float("-inf")

            if temperature > 0:
                probs = F.softmax(logits.float() / temperature, dim=-1)
                x0 = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, -1)
            else:
                x0 = logits.argmax(dim=-1)

            # Confidence score used to rank which predictions to commit this step.
            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                conf = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)   # (B, T)
            elif remasking == "random":
                conf = torch.rand(x.shape, device=device)
            else:
                raise ValueError(f"unknown remasking strategy: {remasking!r}")

            # Only currently-masked positions INSIDE the active block are eligible
            # to be committed. Everything else is excluded with a -inf score.
            eligible = is_mask.clone()
            eligible[:, :s0] = False
            eligible[:, s1:] = False
            conf = torch.where(eligible, conf, torch.full_like(conf, float("-inf")))

            k = int(counts[i])
            if k > 0:
                commit_idx = conf.topk(k, dim=1).indices               # (B, k)
                commit = torch.zeros_like(is_mask)
                commit.scatter_(1, commit_idx,
                                torch.ones_like(commit_idx, dtype=torch.bool))
                x = torch.where(commit, x0, x)

    if was_training:
        model.train()
    return x
