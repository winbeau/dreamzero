# Novelty audit (living document)

Search date: 2026-08-29.  This audit uses public papers and official
repositories.  It is deliberately phrased as a falsifiable boundary, not a
blanket "first sparse WAM" claim.

| Work | Relevant mechanism | Overlap | Boundary from this method |
|---|---|---|---|
| [FAST-AR](https://arxiv.org/abs/2602.01801) | Training-free temporal cache compression plus ANN sparse self/cross attention for AR video diffusion and world models | Long causal KV and semantic sparse attention | Generic video/world-model routing; not grounded in robot camera geometry or conditioned on action queries for closed-loop control |
| [SparsePR](https://arxiv.org/abs/2608.18484) | Executable block-sparse attention with response-coupled partitions and probe-fitted residual reconstruction | Strongest generic executable sparse-attention baseline; 22-26% executed-pair density | General video/world models; no embodied view budget, RGB-token correspondence, or action-conditioned anchor definition |
| [Faster-WAM](https://arxiv.org/abs/2608.04404) | SparseMoT selects a compact subset of network stages for video-action interaction; Interval KV-Fusion reuses future features | Directly establishes sparsity/selection inside WAM inference | Stage/layer-wise future conditioning, not token-wise sparse self-attention over causal visual KV |
| [Efficient-WAM](https://arxiv.org/abs/2606.10040) | Compact video expert, token-sparse video latents, asymmetric video/action denoising | Directly establishes token sparsity in WAMs | Changes the future-imagination representation/model; does not route historical attention KV via action-conditioned, pixel-grounded anchors |
| [LingBot-VA](https://arxiv.org/abs/2601.21998) | Joint causal video-action world model described with dual-stream MoT | Related embodied causal WAM architecture | Official [repository](https://github.com/Robbyant/lingbot-va) currently releases shared-backbone weights/code and explicitly says the separated version is pending; it is not an architecture-aligned primary base |
| [DreamZero](https://arxiv.org/abs/2602.15922) | 14B AR video diffusion WAM with joint action prediction and long causal KV | Primary base and evaluation platform | Upstream uses dense causal video/action attention; our change is an inference operator and routing policy |

## Provisional contribution boundary

The current evidence supports investigating this narrower claim:

> Action-conditioned semantic anchors, made spatially interpretable through a
> CNN-VAE/DiT token-to-RGB correspondence and executed through a cached,
> fixed-shape sparse visual-KV route, can accelerate closed-loop embodied WAM
> inference without degrading control.

The audit does not yet justify "first".  Before submission it must be expanded
with citation-graph traversal, contemporaneous August 2026 searches, code-level
checks of all closest methods, and a claim-by-claim overlap matrix.
