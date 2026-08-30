# Method revision: from short-token Fast-WAM to embodied anchor sparsity

## Decision

The primary speed claim moves from Fast-WAM to the released DreamZero-14B
autoregressive (AR) World Action Model.  Fast-WAM remains a short-sequence
negative control rather than being discarded.

The reason is empirical, not cosmetic.  Fast-WAM's critical joint input has
only 392 tokens (360 visual plus 32 action tokens).  Measured fused
self-attention is 4.872% of `infer_joint`, giving an Amdahl ceiling of only
1.051x even if attention were free.  The optimized sparse path was slower
(0.696x).  DreamZero-14B AR instead uses 880 spatial tokens per latent frame,
1,760 current video-query tokens, and a typical 7,920-token causal video KV
history, so executable KV sparsity can cover enough work to matter.

## Revised scientific story

Embodied video contains a small number of spatial anchor patches that matter
disproportionately for control: the manipulated object, gripper-object contact,
target receptacle, and other task-relevant regions.  DreamZero's convolutional
VAE and convolutional DiT patch embedding preserve a deterministic mapping
between each latent token and a region of the RGB composite.  We exploit this
correspondence to route historical video KV tokens using action queries, while
keeping current action/state keys and both current visual frames dense.

The method is therefore not generic magnitude top-k attention.  Its defining
constraints are:

1. **Embodiment-conditioned:** action queries determine which historical visual
   regions are retained.
2. **Pixel-grounded:** every routed token maps back to an exact RGB patch and a
   known camera view.
3. **View-balanced:** wrist, exterior-left, and exterior-right views each
   receive a fixed share of the route budget.
4. **Executable:** all heads share one fixed-length route, allowing one gathered
   FlashAttention call rather than irregular per-head kernels.
5. **Amortized:** the route is computed once and reused across transformer
   layers and denoising calls for the same causal control block.

## Base-model policy

- **Primary:** DreamZero-14B AR, because the public repository, paper, model
  configuration, action pathway, and long causal video KV cache align.
- **Negative control:** Fast-WAM, demonstrating why sparse attention is not
  automatically beneficial for a short 392-token joint workload.
- **Not primary:** LingBot-VA.  Its paper describes a dual-stream
  Mixture-of-Transformers design, while the public repository currently states
  that only shared-backbone code and weights are released and asks users to
  wait for the separated version.  Claims and implementation experiments must
  not silently mix those two architectures.

## Claim discipline

We do **not** claim that sparse attention is new for video/world models, or that
sparsity is new for WAMs.  The provisional contribution is the narrower
combination of action-conditioned, RGB-grounded spatial anchor routing and a
cached fixed-shape sparse video-KV operator inside a closed-loop embodied WAM.
The claim remains provisional until the full novelty audit and evaluation are
complete.
