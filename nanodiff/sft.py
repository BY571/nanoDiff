"""Supervised fine-tuning (SFT) — teaching the base model to follow instructions.

SFT is a *tiny* change from pretraining (LLaDA Algorithm 2):

  * a training example is a (prompt, response) pair;
  * the forward process masks ONLY the response tokens — the prompt stays clean
    and acts as pure conditioning;
  * the loss is the same 1/t-weighted cross-entropy, computed over masked
    response tokens, normalized by the response length L' (not the full
    sequence length T).

Padding short responses with <|endoftext|> (and keeping those pad tokens in the
loss) is what teaches the model to control its own response length — unlike an
autoregressive model it cannot just stop, so "after the answer, predict EOT"
has to be learned.

This mirrors how `sampler.generate` already works at inference: a clean prompt
prefix, a response region that starts fully masked and gets denoised. SFT just
makes training match that setup.

Reference: LLaDA (arXiv:2502.09992), Section 2.3 and Algorithm 2.
"""
import torch
import torch.nn.functional as F


def sft_forward_process(prompt_ids, response_ids, mask_token_id, t_eps=1e-3, t=None):
    """Corrupt only the RESPONSE span of a (prompt, response) pair.

    The one change from pretraining's `forward_process`: the prompt is pure
    conditioning and is never masked; only the response is corrupted.

    Args:
        prompt_ids:    (B, P) clean prompt token ids — never masked
        response_ids:  (B, L) clean response token ids — the part we corrupt
        mask_token_id: id of the [MASK] / absorbing-state token
        t_eps:         lower clamp for t so the 1/t loss weight stays finite
        t:             optional (B,) pre-sampled mask probabilities; else U(t_eps, 1)

    Returns:
        x_t:           (B, P+L) prompt followed by the corrupted response
        response_mask: (B, P+L) bool, True only at masked response positions
                       (always False over the prompt span)
        t:             (B,) the per-sequence mask probability used
    """
    B, P = prompt_ids.shape
    L = response_ids.shape[1]
    device = prompt_ids.device
    x0 = torch.cat([prompt_ids, response_ids], dim=1)        # (B, P+L)

    if t is None:
        t = torch.rand(B, device=device) * (1.0 - t_eps) + t_eps

    # Draw masks for the whole sequence, then keep them only in the response
    # span — the prompt is never corrupted.
    response_mask = torch.rand(B, P + L, device=device) < t.unsqueeze(1)
    response_mask[:, :P] = False

    x_t = torch.where(response_mask, mask_token_id, x0)
    return x_t, response_mask, t


def sft_loss(logits, response_ids, response_mask, t):
    """SFT training objective — LLaDA Algorithm 2.

    Identical to `diffusion.diffusion_loss` except the loss is computed over the
    RESPONSE span only and the per-sequence normalizer is the response length
    L' (not the full sequence length T).

    Args:
        logits:        (B, P+L, V) model predictions for the full sequence
        response_ids:  (B, L)      clean response targets
        response_mask: (B, P+L)    bool, True at masked response positions
        t:             (B,)        per-sequence mask probability
    Returns:
        scalar loss (a stochastic upper bound on the response NLL).
    """
    B, L = response_ids.shape
    V = logits.shape[-1]
    # The response is the final L positions of the [prompt | response] sequence.
    resp_logits = logits[:, -L:, :]               # (B, L, V)
    resp_mask = response_mask[:, -L:]             # (B, L)

    ce = F.cross_entropy(
        resp_logits.reshape(-1, V),
        response_ids.reshape(-1),
        reduction="none",
    ).reshape(B, L)
    ce = ce * resp_mask                           # masked response positions only
    per_seq = ce.sum(dim=1) / (t * L)             # 1/t weight, normalized by L'
    return per_seq.mean()
