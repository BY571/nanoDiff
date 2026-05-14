"""[STUB] Autoregressive initialization for nanoDiff — a planned learning milestone.

Training a diffusion LM from scratch is expensive. Dream 7B and LLaDA 2.0 instead
*convert* a pretrained autoregressive (AR) model into a diffusion model — inheriting
its learned knowledge and converging far faster than from-scratch training.

Because nanoDiff is already a LLaMA-style transformer, the conversion is mostly
plumbing:

  * load AR transformer weights into the (architecturally identical) bidirectional
    nanoDiff model — if the AR checkpoint is also RMSNorm + SwiGLU + RoPE, the
    weight shapes line up and it is just a key remapping;
  * drop the causal mask — already done, nanoDiff attention is bidirectional;
  * optionally anneal attention from causal -> bidirectional over early training
    so the model adapts gradually rather than seeing a distribution shock;
  * continue training with the masked-diffusion objective for a short "adaptation"
    phase (LLaDA 2.0 uses a 3-phase block-level WSD schedule for this).

References:
  Dream 7B  (arXiv:2508.15487)  — AR-init + context-adaptive noise reschedule
  LLaDA 2.0 (arXiv:2512.15745)  — AR -> diffusion conversion at scale

Implementation sketch
---------------------
    def load_ar_weights(nanodiff_model, hf_model_name_or_path):
        # map AR checkpoint param names -> nanoDiff param names, load_state_dict
        ...

    def attention_anneal_schedule(it, total_anneal_iters):
        # returns a value in [0, 1]: fraction of bidirectional-ness at step `it`
        ...
"""


def load_ar_weights(*args, **kwargs):
    raise NotImplementedError(
        "AR-initialization is a planned milestone — see the docstring and README roadmap."
    )


def attention_anneal_schedule(*args, **kwargs):
    raise NotImplementedError(
        "AR-initialization is a planned milestone — see the docstring and README roadmap."
    )
