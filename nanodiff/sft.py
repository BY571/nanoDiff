"""[STUB] Supervised fine-tuning (SFT) for nanoDiff — a planned learning milestone.

SFT teaches the pretrained base model to follow instructions. The recipe (LLaDA
Algorithm 2) is a *tiny* change to pretraining:

  * a training example is a (prompt, response) pair;
  * the forward process masks ONLY the response tokens — the prompt stays clean;
  * the loss is the same 1/t-weighted cross-entropy, computed over masked response
    tokens, normalized by the response length L' (not the full sequence length).

Padding short responses with <|endoftext|> (and including those pad tokens in the
loss) is what teaches the model to control its own response length.

Reference: LLaDA (arXiv:2502.09992), Section 2.3 and Algorithm 2.

Implementation sketch
---------------------
    def sft_forward_process(prompt_ids, response_ids, mask_token_id, t_eps):
        # concat -> [prompt | response]; sample t ~ U(t_eps, 1);
        # mask only the response span; return x_t, response_mask, t
        ...

    def sft_loss(logits, response_ids, response_mask, t):
        # identical to diffusion.diffusion_loss but the per-sequence normalizer is
        # L' (the response length) instead of T (the full sequence length)
        ...

    # training loop: load instruction pairs, pad responses with <|endoftext|>,
    # then reuse train.py's optimizer / WSD schedule / DDP scaffolding unchanged.
"""


def sft_forward_process(*args, **kwargs):
    raise NotImplementedError(
        "SFT is a planned milestone — see this module's docstring and the README roadmap."
    )


def sft_loss(*args, **kwargs):
    raise NotImplementedError(
        "SFT is a planned milestone — see this module's docstring and the README roadmap."
    )
