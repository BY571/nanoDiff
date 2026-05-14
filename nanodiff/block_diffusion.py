"""[STUB] Block diffusion (BD3-LM) for nanoDiff — a planned learning milestone.

Block diffusion interpolates between autoregressive and diffusion LMs: split the
sequence into blocks, generate blocks left-to-right (autoregressive ACROSS blocks),
and run masked diffusion WITHIN each block. Payoff: KV-caching, arbitrary-length
generation, and a block-size quality/speed knob — while keeping diffusion's
parallel decoding inside a block.

Note: nanodiff/sampler.py ALREADY supports semi-autoregressive *sampling* via the
`block_length` argument. What this module adds is block-aware *training*, so the
model is actually optimized for that decoding regime:

  * a block-causal attention mask — a token attends to all tokens in earlier
    blocks plus all tokens in its own block, but not to future blocks;
  * the BD3-LM loss = sum of per-block masked-diffusion objectives;
  * (inference) a KV cache over already-finalized blocks.

Reference: BD3-LM (arXiv:2503.09573) — Arriola et al., "Block Diffusion",
           ICLR 2025 Oral.

Implementation sketch
---------------------
    def block_causal_mask(seq_len, block_size, device):
        # (seq_len, seq_len) bool mask; True = allowed to attend.
        # Pass it to F.scaled_dot_product_attention as `attn_mask=`.
        ...

    class BlockDiffusionModel(NanoDiff):
        # same model, but forward() accepts/builds a block-causal mask
        ...

    def bd3lm_loss(logits, x0, mask, t, block_size):
        # diffusion_loss applied per block, then summed/averaged
        ...
"""


def block_causal_mask(*args, **kwargs):
    raise NotImplementedError(
        "Block diffusion is a planned milestone — see the docstring and README roadmap."
    )


def bd3lm_loss(*args, **kwargs):
    raise NotImplementedError(
        "Block diffusion is a planned milestone — see the docstring and README roadmap."
    )
