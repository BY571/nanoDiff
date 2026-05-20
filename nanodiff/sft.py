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


# Prompt templates. The trailing "### Response:\n" is the generation cue — at
# inference the model is conditioned on the prompt up to and including it, then
# denoises the response region. Kept deliberately lean (no verbose preamble):
# a 50M model has little capacity to spend on boilerplate.
SFT_PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"
SFT_PROMPT_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def encode_sft_example(instruction, input_text, output, enc,
                       prompt_len, response_len, eot_id):
    """Encode one instruction example into fixed-length (prompt, response) ids.

    The prompt is **right-aligned**: truncated from the LEFT and left-padded
    with `eot_id`, so the "### Response:" cue always lands at the end and the
    response always begins at position `prompt_len`.

    The response is **left-aligned**: an `eot_id` end-marker is appended, then
    it is truncated from the RIGHT / right-padded with `eot_id`. Those pad
    tokens stay in the SFT loss — that is what teaches the model to emit EOT
    and control its own answer length.

    Args:
        instruction:  the task instruction (str)
        input_text:   optional extra context (str; blank -> no Input block)
        output:       the target response (str)
        enc:          a tiktoken encoding
        prompt_len:   fixed prompt length P
        response_len: fixed response length L
        eot_id:       <|endoftext|> id — used for padding and as the end-marker

    Returns:
        (prompt_ids, response_ids) — python int lists of length P and L.
    """
    if input_text and input_text.strip():
        prompt_str = SFT_PROMPT_WITH_INPUT.format(instruction=instruction,
                                                  input=input_text)
    else:
        prompt_str = SFT_PROMPT_NO_INPUT.format(instruction=instruction)

    prompt = enc.encode(prompt_str)
    if len(prompt) > prompt_len:
        prompt = prompt[-prompt_len:]                    # keep the end (the cue)
    else:
        prompt = [eot_id] * (prompt_len - len(prompt)) + prompt

    response = enc.encode(output) + [eot_id]             # EOT marks the answer's end
    if len(response) > response_len:
        response = response[:response_len]               # keep the start
    else:
        response = response + [eot_id] * (response_len - len(response))

    return prompt, response


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
