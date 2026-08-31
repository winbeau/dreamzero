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
