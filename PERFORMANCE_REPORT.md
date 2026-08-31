# Dynamic M1/M2 performance report

Date: 2026-08-31

## Status

This report freezes the Dense and old per-layer gather/scatter sparse DiT
baseline required before Dynamic Packed M2 work and records the first complete
eight-DiT Dynamic Packed M2 service smoke. The 100-request runs, GPU-group
exchange, closed-loop measurements, and final confidence intervals are not
complete, so no final performance claim is made here.

## Protocol invariant

DreamZero retains the released 16 scheduler steps.  With `NUM_DIT_STEPS=8`,
the fixed mask executes eight real DiT evaluations at scheduler indices
`0, 1, 2, 6, 10, 13, 14, 15`.  Sparse results are not allowed to change this
mask or reduce the number of real DiT calls.

Checkpoint baseline setup:

- checkpoint: released DreamZero-DROID, 16.48B diffusion parameters;
- BF16 FlashAttention 2, eager mode;
- seven cached frames plus two current frames;
- 40 layers and 40 heads;
- old sparse configuration: historical/current/Q keep 20%/20%/20%, two recent
  dense KV frames, dense prefix/suffix 5/5, radius-one propagation every five
  middle layers, route reuse, and exact no-update denoising;
- two warmups and six measured Dense/Sparse forwards on physical H200 GPUs
  2, 3, 5, and 6 concurrently;
- every rank performs a paired Dense then Sparse measurement on the same GPU.

## Frozen old sparse DiT baseline

| Physical GPU | Dense p50 | Old Sparse p50 | Speedup |
| ---: | ---: | ---: | ---: |
| 2 | 187.90 ms | 135.66 ms | 1.385x |
| 3 | 189.46 ms | 134.15 ms | 1.412x |
| 5 | 189.69 ms | 135.92 ms | 1.396x |
| 6 | 186.51 ms | 121.54 ms | 1.535x |
| Mean | 188.39 ms | 131.82 ms | 1.432x |

The ratio of the four mean latencies is 1.429x.  GPU-pair means are 1.399x on
2--3 and 1.465x on 5--6.  The spread, especially GPU 6, proves that later
paper measurements must use the required GPU-group exchange rather than quote
the best device.

All four ranks pass:

- full-budget video/action/all-layer-cache exactness;
- update-disabled versus update-enabled Dense video/action exactness;
- update-disabled versus update-enabled Sparse video/action exactness.

The old 20/20/20 approximation has synthetic video/action relative L2 of
87.14%/11.23% on every rank and therefore remains a speed ceiling, not a
quality-selected policy.

Raw artifacts:

`dynamic_m1_m2/baseline/20260830_old_sparse_gpu2356/`

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=2,3,5,6 DREAMZERO_DISABLE_TORCH_COMPILE=true \
  uv run --frozen --no-sync torchrun \
  --nproc_per_node=4 --master_port=29641 \
  benchmarks/validate_dreamzero_checkpoint.py \
  --model-path /data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse/checkpoints/DreamZero-DROID \
  --output-dir /data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/dynamic_m1_m2/baseline/20260830_old_sparse_gpu2356 \
  --physical-gpus 2 3 5 6 \
  --keep-ratios 0.20 0.20 0.20 0.20 \
  --current-keep-ratios 0.20 0.20 0.20 0.20 \
  --attention-query-keep-ratios 0.20 0.20 0.20 0.20 \
  --dense-prefix-layers 5 --dense-suffix-layers 5 \
  --propagate-radius 1 --propagate-every 5 \
  --reuse-denoise --current-attention --no-update-kv-cache \
  --warmup 2 --repeats 6
```

## Stage evidence

- baseline code revision: `d2999f64a8143285a26fc09f84d241866770948c`;
- raw baseline completed on GPUs 2/3/5/6;
- the fixed Packed M2 row remains pending and must be added before stage 1 is
  considered complete.

## Preliminary Dynamic Packed M2 service result

The `timestep_segment_balanced` candidate was measured through the real
WebSocket service with one warmup and three paired measured requests. Dense ran
on GPUs 2--3 and Sparse on GPUs 5--6. The client inputs, seed, scheduler, and
number of DiT calls were held fixed.

| Metric | Dense | Sparse | Speedup |
| --- | ---: | ---: | ---: |
| client end-to-end mean | 2.0616 s | 1.5078 s | 1.367x |
| client p50 | 1.8456 s | 1.4655 s | 1.259x |
| client p90 | 2.3995 s | 1.6160 s | 1.485x |
| server inference mean | 2.0507 s | 1.4940 s | 1.373x |
| diffusion mean | 1.4800 s | 1.0767 s | 1.375x |

All three measured pairs were faster under Sparse. The paired geometric mean
speedup was 1.353x, with an intentionally non-claimable three-sample bootstrap
95% interval of [1.089x, 1.732x]. Server logs record `DIT Compute Steps 8
steps` for every warmup and measured request in both arms.

The preliminary full-service mean clears the 1.35x target, but the Packed DiT
target, 100-request sample size, three GPU exchanges, confidence lower bound,
and 95%-faster-request gate remain open.

Raw artifacts:

`dynamic_m1_m2/e2e/20260830_balanced_smoke/`

## Real DROID diagnostic performance

The first task-disjoint DROID history-chain smoke used one episode at its
early, middle, and late trajectory stages, with three history calls before
each target. One target was reserved as warmup and two were measured. This is
too small for a paper latency claim but prevents selection on random images.

| Packed policy | Mean target latency | Speedup vs Dense | Quality decision |
| --- | ---: | ---: | --- |
| Dense | 1.9032 s | 1.000x | reference |
| balanced | 1.3393 s | 1.421x | reject |
| quality | 1.3414 s | 1.419x | reject |
| quality + history floor 75% | 1.6455 s | 1.157x | reject |
| quality + Dense history | 1.5057 s | 1.264x | reject |
| current floor 75% + Dense history | 1.6088 s | 1.183x | reject late target |
| full-budget Packed | 1.9379 s | 0.982x | exactness control |

The nominal 1.42x rows cannot be reported as accepted acceleration because
they fail action quality. The conservative rows show that indiscriminately
buying quality with shared history/current floors also erases too much of the
speed target. The performance path now depends on confident per-request or
shared-group sparse routing plus Dense fallback, not a single global table.

## Expanded DROID performance and profile ceiling

The real-history replay now covers 108 request keys across the immutable
train/validation/test split. Balanced Packed M2 measures 1.453x, 1.476x, and
1.493x mean end-to-end speedup on the three splits; the 75%-current/Dense-
history profile measures 1.191x, 1.194x, and 1.173x. Every request in both
global sparse arms is faster than its Dense pair and every call retains eight
real DiT evaluations.

These are performance ablations, not accepted policies, because their action
quality fails. A request-level Oracle choosing the fastest quality-safe arm
from balanced/conservative/Dense reaches only 1.1085x train, 1.0683x
validation, and 1.1254x test speedup. The present global-profile family cannot
meet the 1.35x target regardless of classifier accuracy.

The first-two-DiT-Dense table reaches 1.373x validation and 1.425x test mean
speedup with all 36 requests faster, but fails action quality on 32 of 36
requests. It is rejected rather than promoted to the 100-request timing run.

The online flow sentinel record-only test keeps exactly eight model calls. Its
validation-frozen threshold is rejected on safety, so the optional Dense
recomputation path is excluded from the main performance result; it would add
an estimated 1.33 model calls per request.

## Heterogeneous current-QKV performance ablation

The two-group executor performs real channel-sliced current Q/K/V/O and one
packed varlen FA2 launch. It reduces the matched old two-group attention
microbenchmark from 2.20 to 2.11 ms, but a regular fixed-shape 40-head call is
still 1.54 ms. On the released early timestep, 50% and 35% outer trunks reach
only 1.063x and 1.317x p50 DiT speedup. The most favorable final warmed 50%
sample is 135.73 ms versus 187.52 ms Dense, or 1.38x. Neither row is promoted
to the 100-request run.

Artifact:

`dynamic_m1_m2/dynamic_budgets/20260831_two_group_qkv/checkpoint_early_gpu01_varlen_lowtrunk/`

## Dense action-history performance ablation

The 25 action/state queries were separately evaluated against complete
historical K/V while video queries retained the balanced sparse history. On
the same 14B checkpoint input this raises Sparse p50 from 154.98 to 164.43 ms
and reduces paired DiT speedup from 1.228x to 1.147x. The corresponding
single-layer attention microbenchmark measures 0.603 ms extra work, about 23
ms over 38 packed layers without overlap.

The task-disjoint validation replay contains 18 targets, each preceded by
three real history calls. All 72 service calls retain exactly eight DiT model
calls and no Dense rerun. Relative to the existing Dense 2--3 replay:

| Metric | Dense | Balanced + Dense action history | Result |
| --- | ---: | ---: | ---: |
| mean target latency | 1.9034 s | 1.4292 s | 1.332x |
| p50 target latency | 1.9107 s | 1.4036 s | 1.361x |
| paired geometric-mean speedup | -- | 1.3329x | CI95 [1.3058x, 1.3539x] |
| Sparse-faster requests | -- | 18 / 18 | 100% |

The original balanced validation replay measured 1.2899 s and 1.476x mean
speedup. Protecting action history therefore costs 10.8% Sparse latency and
drops below the 1.35x end-to-end mean target while still failing quality. It
is not promoted.

Artifact:

`dynamic_m1_m2/e2e/20260831_dense_action_history_balanced_validation18/`

## Dynamic action-history layer schedule

The fixed 8x40 action-history table restores most of the lost latency by
protecting only early Transformer layers. At the same early-DiT checkpoint,
layers 1--13 measure 154.57 ms Sparse p50 and 1.217x speedup, versus 157.91 ms
and 1.202x for layers 28--38, and 164.43 ms and 1.147x for all 38 packed
layers.

On validation18, early-layer protection measures 1.3575 s mean target latency
against 1.9034 s Dense: 1.402x ratio-of-means, 1.408x paired geometric mean,
CI95 [1.345x, 1.461x], and 18/18 Sparse-faster requests. All 72 history/target
calls retain exactly eight DiT model calls. The performance row clears the
mean target but is rejected because final-action quality fails every request.

At DiT index 4, early-layer protection costs 13.28 ms versus the same dynamic
budget with no action-history protection (145.59 versus 132.31 ms), reducing
paired DiT speedup from 1.417x to 1.308x. The measured quality gain is real,
but a global action-readout fix cannot meet both trajectory quality and the
Packed DiT target.

Artifact:

`dynamic_m1_m2/e2e/20260831_dynamic_action_history_early_layers_validation18/`

## Early video-history performance gate

The 75% history floor on layers 1--13 measures 1.2796 s mean target latency
against 1.9034 s Dense, or 1.487x ratio-of-means and 1.489x paired geometric
mean speedup. CI95 is [1.452x, 1.512x], all 18 targets are faster, and all 72
history/target calls retain eight DiT model calls. It is rejected solely on
quality; no performance claim treats this profile as accepted.

## Dense-suffix recovery performance

Balanced Packed M2 was replayed with three and five Dense suffix layers on
auxiliary GPUs 0--1 against the unchanged Dense GPUs 2--3 reference.  Each row
uses one independent four-call warmup chain followed by 18 measured targets,
each with three real history calls.  Server logs contain 76 entries with eight
DiT model evaluations for each row.

| Dense suffix | Sparse mean | Ratio-of-means | Paired geomean | CI95 | Faster |
| ---: | ---: | ---: | ---: | --- | ---: |
| 3 | 1.2857 s | 1.480x | 1.484x | [1.431x, 1.518x] | 18/18 |
| 5 | 1.3425 s | 1.418x | 1.420x | [1.377x, 1.457x] | 18/18 |

Suffix three retains the end-to-end stretch target but fails quality.  Suffix
five spends 4.4% more Sparse latency and also fails quality, so deeper output
recovery is not promoted to the 100-request experiment.

Artifacts:

```text
dynamic_m1_m2/e2e/20260831_balanced_suffix3_validation18/
dynamic_m1_m2/e2e/20260831_balanced_suffix5_validation18/
```

## Radius-three propagation performance

The promoted radius-three/every-five service uses the original suffix-one
balanced table.  Against the unchanged Dense validation18 reference it
measures 1.2563 s mean target latency, 1.515x ratio-of-means speedup, and 1.516x
paired geometric mean speedup with CI95 [1.485x, 1.542x].  Every target is
faster, the minimum paired speedup is 1.351x, and all 76 service calls contain
exactly eight DiT evaluations.

This is the fastest validation18 row in the current propagation family, but it
is not an accepted performance claim because action quality fails 18/18.  The
executor should not trade on the favorable radius timing until an
action-sensitive recovery path passes quality.

Artifact:

`dynamic_m1_m2/e2e/20260831_balanced_r3e5_validation18/`

## Maximum-current action-readout performance gate

The scheduled maximum-current experiment was run twice with segment-entry and
segment-exit candidates exchanged across auxiliary GPUs 0 and 1. Each rank
uses the same balanced 8x40 budget, radius two/every five propagation, suffix
one, five recorded forwards after two warmups, and no KV-cache update.

| Candidate | GPU0 speedup | GPU1 speedup | Exchanged geomean |
| --- | ---: | ---: | ---: |
| segment entries | 1.298x | 1.228x | 1.26235x |
| segment exits | 1.278x | 1.247x | 1.26225x |

The order reversal confirms a stable GPU throughput difference and removes
the apparent first-round advantage of the entry schedule. Both candidates
perform eight extra 25-query current-K/V readouts and have indistinguishable
aggregate speed. A same-commit two-rank `none` control measures 1.285x and
1.127x, but is not used to claim negative overhead because independent loads
show large first-use compilation samples and material p50 drift. Extra K/V
projection cannot logically accelerate the executor; a longer within-process
timing design would be required to resolve its small cost.

The entry schedule is not promoted to end-to-end validation because its only
reproducible quality improvement is 0.00387 action-L2 percentage points. The
global trajectory error remains approximately 9%, so spending a validation18
run on this readout cannot change the quality gate or the main performance
claim.

Artifact:

`dynamic_m1_m2/dynamic_budgets/20260831_max_action_current/`
