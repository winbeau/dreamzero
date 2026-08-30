# Embodied Anchor Sparse Attention

## Geometry

The released DreamZero-DROID preprocessing builds a 352x640 three-camera RGB
composite:

- wrist camera: the full top half;
- exterior-left camera: bottom-left quadrant;
- exterior-right camera: bottom-right quadrant.

The Wan VAE downsamples space by 8x, producing a 44x80 latent.  The DiT
`Conv3d` patch embedding uses a `(1, 2, 2)` kernel and stride, producing a 22x40
token grid, or 880 tokens per latent frame.  Consequently, each DiT token maps
to one exact 16x16 RGB patch.  This mapping is implemented and tested in
`embodied_anchor_sparse.py`.

## Router

Let action queries be `Q_a` and the causal video KV history be `K_v, V_v`.
Using a small prefix of attention heads and head dimensions, the router scores
each video token by its maximum normalized similarity to any action token:

`s_j = max_(a,h) <normalize(Q_a[a,h]), normalize(K_v[j,h])>`.

Scores are averaged over a local spatial neighborhood and partitioned by
camera view.  Within each old frame, the router keeps an exact, view-balanced
budget.  The newest `r` frames are retained densely.  For released
DreamZero-DROID, `r=2` preserves the complete current two-frame generation
block.  Action and state KV tokens are always dense.

For a history of `F` frames, per-frame anchor budget `k`, and `r` recent dense
frames, the executed video-KV length is:

`(F-r) * k + r * 880`.

The selected indices are shared by every attention head.  Keys and values are
gathered once per layer and evaluated by one dense FlashAttention kernel over
the shorter sequence.

## Route lifetime

Recomputing a top-k route in every layer is counterproductive.  A first H200
gate at 25% old-frame keep ratio showed:

- dense attention: 1.096 ms p50;
- gathered sparse attention: 0.724 ms p50;
- router construction: 0.661 ms p50;
- router + sparse attention when recomputed per call: 1.459 ms p50 (0.751x).

The implementation therefore computes the route in transformer layer 0 and
passes only its token indices through all later layers.  The model also caches
those indices across denoising calls with the same causal control-block key.
The cache is cleared on causal rewind and can be explicitly cleared at episode
reset with `clear_anchor_sparse_route_cache()`.

## Initial H200 cached-route gate

The first uncontended operator gate used the same DreamZero video geometry
(`B=1`, 40 heads, head dimension 128, 1,760 video queries, 7,920 video KV
tokens) with a 32-action/no-state proxy instead of the released checkpoint's
24 action plus one state register tokens.
PyTorch BF16 Flash SDPA on an NVIDIA H200 NVL produced:

| Old-frame keep | Recent dense | Executed video KV | Gather + attention p50 | Amortized speedup over dense |
|---:|---:|---:|---:|---:|
| 10% | 1 frame | 20.0% | 0.376 ms | 2.83x |
| 15% | 1 frame | 24.4% | 0.451 ms | 2.37x |
| 20% | 1 frame | 28.9% | 0.528 ms | 2.03x |
| 25% | 1 frame | 33.3% | 0.587 ms | 1.83x |

The amortized number charges one route construction across 40 layers and only
one denoising call, so it is conservative when denoising reuse is enabled.  The
table uses one recent dense frame and is therefore an aggressive operator upper
bound rather than the primary method configuration.  The primary configuration
keeps both current frames dense; at 20%/25% old-frame keep ratio this executes
37.8%/41.7% of a nine-frame video KV history.

An uncontended exact-shape FlashAttention 2 rerun was then repeated on physical
H200 GPUs 2--5 with 300 samples per GPU, 1,785 queries (1,760 video, 24 action,
one state), and 7,920 video KV tokens.  The table reports the median across the
four per-GPU p50 measurements:

| Old-frame keep | Recent dense | Gather + FA2 p50 | Router p50 | Router/40 + attention | Speedup |
|---:|---:|---:|---:|---:|---:|
| 20% | 2 frames | 0.681 ms | 0.629 ms | 0.697 ms | 1.67x |
| 25% | 2 frames | 0.725 ms | 0.637 ms | 0.741 ms | 1.57x |

These are operator-level gates, not end-to-end or quality results.  The 20% and
25% configurations advance as the initial speed-quality candidates; model-level
full-budget parity, task quality, and end-to-end timing remain mandatory.

## Safety and parity invariants

- The feature is disabled by default.
- Version 1 activates only for the released DROID 22x40/880-token layout.
- A keep ratio of 1.0 bypasses routing/gathering and follows the original dense
  path.
- Cached routes are validated by batch, device, causal position, KV length, and
  fixed selected length.
- The route never changes the full KV cache; sparsity affects only attention
  execution, preserving future cache updates and easy dense fallback.
