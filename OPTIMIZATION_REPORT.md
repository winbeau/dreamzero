# DreamZero anchor-sparse optimization report

Date: 2026-08-30

## Executive result

Exact removal of unused action-denoise KV-cache outputs and denoise-persistent
historical anchor K/V reuse improves the selected 0.20/0.50 checkpoint DiT
speedup from about 1.240x to 1.325x without changing dense or sparse outputs.

An aggressive search requested after that gate tested 80% and 90% sparsity for
both historical KV and current-token computation, including sparse current
self-attention queries.  The 80/80 configuration reaches 1.353x mean and
1.374x P50 end-to-end speedup on the same physical H200 pair, with a paired
95% confidence interval of [1.318x, 1.387x].  It passes the latency target, but
its synthetic video/action relative L2 rises to 87.14%/11.23%.  The 90/90
configuration is a slightly faster DiT microbenchmark but a noisier/slower
end-to-end run and an even worse 93.30%/14.47% numerical proxy.

The aggressive configurations are therefore speed-ceiling ablations, not the
quality-selected default.

A follow-up query sweep confirms that current-video Q can be routed
independently from historical KV and current cross-attention/FFN compute.
However, reducing self-attention Q alone below the 20% current-compute route
does not improve the real DiT: small-query FlashAttention shapes plus the
additional gather/scatter path offset the saved attention FLOPs.

## Revisions and setup

- exact action-denoise implementation: `85a1ca6`;
- strict no-update checkpoint gates: `cfce1a1`;
- documented profile follow-up: `7aad4f2`;
- checkpoint: released DreamZero-DROID revision
  `96ad344138c66e82536422432ad742f015784942`;
- physical GPUs: H200 NVL 5--6, used sequentially for all three end-to-end
  modes;
- environment: repository uv `.venv`, BF16 FlashAttention 2, eager mode with
  `DREAMZERO_DISABLE_TORCH_COMPILE=true`;
- workload: two warmup and 20 measured requests, seed `20260830`, three
  deterministic RGB views, persistent per-run session, 8 real DiT evaluations
  per request;
- dense and sparse services use the same exact no-update optimization because
  action denoising discards updated KV in both modes.

The first freshly loaded service run is retained, but the predeclared hot
repeat is used for matched steady-state comparison, consistent with the
baseline protocol.  All large artifacts remain outside Git.

## Exact action-denoise optimization

The action head marks causal cache-fill calls with `update_kv_cache=True` and
all action-denoise calls with `False`.  Previously the latter still performed
the following work in every transformer layer and then discarded it:

1. concatenate the full historical K/V with current K/V;
2. gather historical anchors again on every denoise evaluation;
3. stack the complete updated K/V tensor for return;
4. append all 40 returned caches to a Python list.

The optimized path propagates the update flag through the model, skips cache
stack/list construction when false, and caches gathered historical sparse K/V
using immutable history and route identities.  Dynamic current K/V is still
computed and action/state tokens remain dense.  Cache-producing rollout
updates retain the original path.

Strict DreamZero-14B gates on both GPUs report exact equality for:

- update-disabled versus update-enabled dense video and action;
- update-disabled versus update-enabled sparse video and action;
- full-budget sparse versus dense video, action, and all 40 layer caches.

For the quality-oriented 0.20 historical / 0.50 current configuration:

| Metric | Original profile mean | Exact optimized mean | Change |
| --- | ---: | ---: | ---: |
| Dense DiT p50 | 195.93 ms | 190.11 ms | -2.97% |
| Sparse DiT p50 | 158.03 ms | 143.53 ms | -9.17% |
| Per-GPU speedup | 1.240x | 1.325x | +6.82% |

Raw files:

- `real_model_gate/20260830_no_kv_update_exact_gpu56/`;
- `real_model_gate/20260830_no_kv_update_gpu56/`.

## Aggressive 80% and 90% sparsity search

Both aggressive configurations use five dense prefix and suffix layers,
radius-one propagation every five middle layers, dense action/state registers,
and two recent dense KV frames.  Sparse current self-attention is enabled, so
the current keep ratio applies before self-attention as well as to later
cross-attention/FFN computation.

| Configuration | Historical KV route | Current video queries | Checkpoint DiT p50 | DiT speedup | Video rel. L2 | Action rel. L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 80/80, keep 0.20/0.20 | 2992 / 7920 | 352 / 1760 | 121.89 ms | 1.540x | 87.14% | 11.23% |
| 90/90, keep 0.10/0.10 | 2376 / 7920 | 176 / 1760 | 118.34 ms | 1.580x | 93.30% | 14.47% |

The 90/90 shape saves only another 3.55 ms per DiT relative to 80/80.  Across
eight DiT evaluations this is about 28 ms/request, small enough for unchanged
encoder/KV work and normal tail latency to dominate an individual end-to-end
run.

Raw checkpoint files are under
`real_model_gate/20260830_aggressive_80_90_gpu56/`.

## Same-hardware 20-request pilot

Hot-repeat raw files are under
`e2e_server/20260830_aggressive_search/`:

- `dense_exact_gpu56_hot_20.json`;
- `sparse_80_80_attn_gpu56_hot_20.json`;
- `sparse_90_90_attn_gpu56_hot_20.json`;
- `paired_dense_vs_80_80_hot_20.json`;
- `paired_dense_vs_90_90_hot_20.json`.

| Mode | Mean | P50 | P90 | Mean speedup | P50 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense exact | 1.8164 s | 1.8113 s | 1.9372 s | 1.000x | 1.000x |
| Sparse 80/80 | 1.3428 s | 1.3186 s | 1.4469 s | 1.353x | 1.374x |
| Sparse 90/90 | 1.4009 s | 1.3972 s | 1.5537 s | 1.297x | 1.296x |

Paired statistics:

| Configuration | Geometric mean | 95% CI | Faster requests | Range |
| --- | ---: | ---: | ---: | ---: |
| 80/80 | 1.353x | [1.318x, 1.387x] | 20 / 20 | 1.237x--1.463x |
| 90/90 | 1.299x | [1.249x, 1.349x] | 20 / 20 | 1.022x--1.485x |

The 80/80 pilot passes the mean, P50, confidence-interval, and faster-fraction
performance gates.  The 90/90 pilot narrowly misses the 1.30x mean target and
has a much wider tail despite its faster isolated DiT.  This is evidence that
maximal token sparsity is not the end-to-end optimum.

## Three-seed 100-request control

The 80/80 configuration was then evaluated with three independent fixed seeds,
100 measured requests per seed, on the same physical GPUs 5--6.  Dense and
sparse services were run sequentially, and all 300 requests were paired by
seed and request index.

| Seed | Dense mean | Sparse mean | Mean speedup | Dense P50 | Sparse P50 | P50 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260830 | 1.7842 s | 1.4012 s | 1.273x | 1.7757 s | 1.3492 s | 1.316x |
| 20260831 | 1.7784 s | 1.3733 s | 1.295x | 1.7763 s | 1.3461 s | 1.320x |
| 20260832 | 1.7768 s | 1.3827 s | 1.285x | 1.7672 s | 1.3476 s | 1.311x |

Combined 300-pair statistics:

- dense/sparse mean: 1.7798 s / 1.3857 s, or 1.284x;
- dense/sparse P50: 1.7750 s / 1.3475 s, or 1.317x;
- dense/sparse P90: 1.8928 s / 1.4702 s, or 1.287x;
- paired geometric-mean speedup: 1.285x;
- paired bootstrap 95% CI: [1.275x, 1.295x];
- sparse faster fraction: 300 / 300;
- paired speedup range: 1.055x--1.459x.

The larger control shows that the 20-request mean estimate was optimistic.
P50, confidence-interval, and faster-fraction gates remain strong, but the
1.284x mean misses the 1.30x paper target by about 1.2 percentage points.  No
requests or seeds are excluded.

Raw files and the aggregate comparison are under
`e2e_server/20260830_100x3/`, including
`paired_dense_vs_80_80_all_300.json`.

## Independent current-query compression

Commit `4e8cc89` separates three budgets that were previously coupled:

1. historical video K/V keep ratio;
2. current-video cross-attention/FFN compute keep ratio;
3. current-video self-attention Q keep ratio.

Action and state queries remain dense because their outputs feed the action
decoder directly.  When the Q and current-compute ratios match, the exact same
route tensor is reused, preserving the original implementation and avoiding a
second router invocation.

The first sweep kept historical KV at 20% while coupling current compute and Q
at progressively smaller ratios:

| Historical KV | Current compute / Q | Q tokens | DiT p50 | Speedup | Video rel. L2 | Action rel. L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 10% / 10% | 176 / 1,760 | 120.44 ms | 1.557x | 93.75% | 13.16% |
| 20% | 5% / 5% | 88 / 1,760 | 119.25 ms | 1.568x | 96.89% | 14.26% |
| 20% | 2.5% / 2.5% | 44 / 1,760 | 134.21 ms | 1.419x | 98.62% | 14.95% |

The 2.5% point is slower rather than faster, demonstrating a real kernel-shape
floor rather than monotonic scaling with token count.  Raw files are under
`real_model_gate/20260830_q_sweep_k20_q10_q5_gpu56/` and
`real_model_gate/20260830_q_sweep_k20_q2p5_gpu5/`.

The decisive sweep then held current cross-attention/FFN compute at 20% and
changed only self-attention Q:

| Historical KV | Current compute | Self-attention Q | Q tokens | DiT p50 | Speedup | Video rel. L2 | Action rel. L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 20% | 20% | 352 / 1,760 | 121.89 ms | 1.540x | 87.14% | 11.23% |
| 20% | 20% | 10% | 176 / 1,760 | 123.24 ms | 1.528x | 87.06% | 12.19% |
| 20% | 20% | 5% | 88 / 1,760 | 122.74 ms | 1.523x | 86.99% | 12.96% |

Therefore the extra speed in the coupled 5% experiment comes from skipping
more cross-attention/FFN token updates, not from Q compression itself.  Q-only
compression is retained as a clean paper ablation and configurable mechanism,
but 20% Q remains the systems choice for the 80/80 speed-ceiling candidate.
Raw decoupled results are under
`real_model_gate/20260830_decoupled_q_k20_compute20_q10_q5_gpu56/`.

## Decision

- Promote the exact no-update/history-KV optimization for all configurations.
- Keep 0.20/0.50 without sparse current self-attention as the conservative
  implementation default until task quality is measured.
- Retain 80/80 with sparse current self-attention as the primary speed-ceiling
  ablation because it is the fastest stable end-to-end configuration tested,
  while noting that its 300-request mean is 1.284x rather than the 1.30x target.
- Retain 90/90 as an ablation showing diminishing systems returns and severe
  numerical degradation; do not use it as the main candidate.
- Retain independent Q routing as an ablation, but do not reduce Q below the
  current-compute route in the primary configuration: it is slower and worsens
  the action numerical proxy.
- Advance both the conservative candidate and 80/80 ablation to matched
  open-loop/closed-loop quality evaluation before making a final paper claim.
