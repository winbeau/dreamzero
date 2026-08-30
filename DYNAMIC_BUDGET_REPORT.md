# Dynamic timestep/layer/head budget report

Date: 2026-08-30

## Status

The fixed-bucket timestep/layer runtime and the one-gather Packed Middle Stack
integration are implemented and pass a released DreamZero-DROID 14B checkpoint
gate. Real timing changes with the selected DiT and layer budgets, while the
full-budget path remains exactly equal to Dense for video, action, and every
returned layer KV cache.

The complete phase remains open. The checkpoint evidence rejects a single
shared budget for all 40 heads, even when the early-DiT middle-stack budget is
raised to 87.5%. A four-bucket M1 head-group executor is now implemented, but
its extra 3--4 FA2 calls per layer recover neither video quality nor enough
latency; this is retained as a negative ablation. Segmented spatial propagation
inside the packed stack is substantially more effective. The current best
early-DiT gate uses radius two every five layers, reaches 0.8987 video cosine at
1.742x DiT speedup, and preserves the action gate. Confidence-controlled M1
runtime routing and a full eight-DiT policy replay are still required before
this phase can pass its quality gate.

Implementation commits:

- `28bbf47`: fixed-shape 8-by-40 dynamic budget table and Packed M2 runtime;
- `6fe77ec`: reproducible Oracle-ordered fixed/timestep/layer/joint ablations;
- `aa9706f`: per-rank early/late checkpoint gate and actual token accounting;
- `e7d8c2d`: align the checkpoint gate with the eight real diffusion timesteps;
- `dff03a7`: first fixed-membership dynamic packed head-group executor;
- `0aa67eb`: group heads per `(timestep, layer)` by shared M1 budget bucket;
- `45db7d4`: segmented packed spatial propagation without per-layer full scatter;
- `71b1a58`: reproducible per-rank propagation sweep and boundary accounting.

All listed commits are pushed to
`origin/codex/dreamzero-anchor-sparse-opt`, and the H200 checkout is
fast-forwarded through `71b1a58`.

## Runtime contract

`DynamicPackedBudgetTable` contains exactly eight real DiT evaluations by 40
Transformer layers. Both historical and current-token budgets must be one of
`[10, 20, 25, 35, 50, 75, 100]%`. Arbitrary per-call shapes are rejected.

The action head sets the runtime DiT index only before an actual DiT
evaluation. The fixed 16-step scheduler still executes eight real DiT calls at
scheduler indices `[0, 1, 2, 6, 10, 13, 14, 15]`; dynamic sparsity does not
reduce this count. The corresponding released-model timesteps are
`[999, 986, 972, 892, 749, 535, 416, 249]`.

Within one DiT evaluation:

- the Dense prefix produces the action-conditioned anchor ordering;
- current tokens are gathered once at the largest middle-layer budget;
- lower budgets use nested prefixes of the same ordering;
- historical KV indices are cached once per ratio bucket;
- route/profile state is reused across layers and denoise evaluations;
- the 24 action tokens and one state token stay Dense;
- RoPE is gathered from every token's original frame/row/column position;
- the packed state is scattered only at the Dense suffix/output boundary.

If every middle layer has a 100% budget, the model bypasses the packed path.
The full-budget invariant is checked independently rather than inferred from
the sparse result.

## Oracle-ordered ablation tables

The table builder consumes only the aggregated Dense Oracle timestep and layer
summaries. It preserves their empirical sensitivity order and assigns a small
fixed bucket distribution. Timestep and layer scores are standardized before
the joint rank assignment because the monotonic timestep effect is smaller
than the U-shaped layer effect.

These are controlled ablation schedules, not substitutes for the calibrated
M1 classifier.

| Profile / ablation | Mean budget | Bucket counts over 8 x 40 cells |
| --- | ---: | --- |
| quality fixed matched | 75.00% | 75%: 320 |
| quality timestep only | 75.00% | 50%: 80, 75%: 160, 100%: 80 |
| quality layer only | 73.75% | 50%: 96, 75%: 144, 100%: 80 |
| quality timestep + layer | 73.75% | 50%: 96, 75%: 144, 100%: 80 |
| aggressive fixed matched | 35.00% | 35%: 320 |
| aggressive timestep only | 38.13% | 20%: 120, 35%: 80, 50%: 80, 75%: 40 |
| aggressive layer only | 37.50% | 20%: 112, 35%: 96, 50%: 80, 75%: 32 |
| aggressive timestep + layer | 37.50% | 20%: 112, 35%: 96, 50%: 80, 75%: 32 |

The equal overall bucket counts for layer-only and joint tables do not mean
their assignments are equal. The joint table redistributes the same matched
compute across both axes.

Table artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/dynamic_budgets/20260830_oracle_ordered/
    quality/
    aggressive/
```

## Released-checkpoint dynamic executor gate

Protocol:

- released DreamZero-DROID 14B AR checkpoint in BF16 on H200 NVL;
- seven historical frames, two current frames, and 25 Dense registers;
- 1/1 Dense prefix/suffix and 38 dynamic packed middle layers;
- same Dense and Sparse input at the selected real diffusion timestep;
- one warmup and five measured forwards;
- no KV-cache mutation inside the timed forward;
- per-rank paired Dense/Sparse timing on auxiliary GPUs 0 and 1.

This is a deterministic real-checkpoint executor and sensitivity gate at the
released DROID geometry. It is not yet the required multi-request, eight-DiT
WebSocket or closed-loop quality result.

| Profile | DiT / timestep | Middle mean budget | Budget range | Dense p50 | Sparse p50 | Speedup | Action cosine | Action rel-L2 | Video cosine | Video rel-L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aggressive joint | early / 999 | 49.34% | 20--75% | 190.04 ms | 106.39 ms | 1.786x | 0.999832 | 1.835% | 0.4897 | 144.21% |
| aggressive joint | late / 249 | 20.79% | 20--35% | 190.00 ms | 80.32 ms | 2.366x | 0.999873 | 1.846% | 0.3558 | 210.76% |
| quality joint | early / 999 | 87.50% | 75--100% | 188.06 ms | 158.71 ms | 1.185x | 0.999899 | 1.454% | 0.8963 | 48.68% |
| quality joint | late / 249 | 52.63% | 50--75% | 189.69 ms | 108.05 ms | 1.755x | 0.999940 | 1.170% | 0.5155 | 145.99% |

The maximum packed shape changes from 1,345 queries by 5,965 keys/values for
the aggressive early step to 641 by 2,797 for its late step. The measured
latency changes accordingly. This establishes that the executor consumes the
dynamic table rather than merely recording it.

All four rows pass the local action cosine >=0.999 and action relative-L2 <=5%
gates. None passes a useful video-output gate. Even the 87.5% early schedule
has 48.68% video relative-L2 and only 1.185x speedup.

For every rank and profile, the independent full-budget run reports:

- video exactly equal to Dense;
- action exactly equal to Dense;
- every returned layer KV cache exactly equal to Dense.

Gate artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/dynamic_budgets/
    20260830_aggressive_tl_early_late_gpu01_v2/
    20260830_quality_tl_early_late_gpu01/
```

The preceding `20260830_aggressive_tl_early_late_gpu01` directory is retained
as an executor-only trace: it varied the budget index while leaving the
synthetic checkpoint timestep fixed at 750. It is excluded from the table
above and from quality claims.

## M1 head-group executor gate

The calibrated M1 prior is quantized to four executor buckets
`[25, 50, 75, 100]%`. Heads are regrouped independently for each
`(timestep, layer)` cell, so a critical Dense head does not force unrelated
heads into the Dense group. Group membership, gathered historical KV, and
indices are cached. Current Q/O/FFN still follow the timestep/layer packed
budget, while each historical-KV group uses one fixed-shape FA2 call. The 25
action/state registers remain Dense in every group.

The leakage-safe prior table has mean historical-KV budget 76.17%, with 37.59%
of head cells assigned Dense. It uses 3--4 nonempty groups in typical layers.
The first fixed-membership design was rejected before promotion because taking
the maximum budget within a static group made more than 85% of effective group
cells Dense; `0aa67eb` replaces it with dynamic shared-ratio membership.

| M1 head groups + aggressive current Q | DiT / timestep | Dense p50 | Sparse p50 | Speedup | Action cosine | Action rel-L2 | Video cosine | Video rel-L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| four dynamic groups | early / 999 | 191.21 ms | 163.79 ms | 1.167x | 0.999821 | 1.902% | 0.4920 | 144.13% |
| four dynamic groups | late / 249 | 187.19 ms | 116.79 ms | 1.603x | 0.999702 | 2.497% | 0.3578 | 210.74% |

This negative result isolates two issues. Historical-KV head grouping does not
repair error caused by dropping current video queries, and launching 3--4 FA2
calls per layer largely removes the packed-kernel speed advantage. The result
does not reject head-dependent M1 budgets; it rejects this particular
historical-KV-only multi-call realization as the final fast path. Its exact
full-budget video/action/cache invariant still passes.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/
    m1_classifier/20260830_full_v2_calibrated/
      selected_m1_bundle_v2_portable.joblib
    dynamic_budgets/
      20260830_m1_head_groups/prior_mean_group4.json
      20260830_m1_headgroup_aggressive_tl_early_late_gpu01/
```

## Segmented packed spatial propagation gate

At a propagation boundary, the executor reconstructs the maximum nested anchor
prefix updated anywhere in the preceding segment, computes its delta, and
propagates that delta over the original frame/row/column grid with a local
spatial average. Selected anchors and all action/state registers remain exact.
The result is immediately repacked for the next segment; the executor does not
return to a full-sequence representation at every Transformer layer. Route,
ordering, RoPE positions, and preallocated buffers are reused.

The following rows all use the aggressive joint current/history budget at the
real early timestep 999. They compare propagation shape/frequency while holding
the 38-layer packed interval and actual DiT count fixed.

| Propagation | Boundaries | Dense p50 | Sparse p50 | Speedup | Action cosine | Action rel-L2 | Video cosine | Video rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| radius 1 / every 5 layers | 8 | 188.02 ms | 112.23 ms | 1.675x | 0.999837 | 1.970% | 0.7719 | 75.47% |
| radius 1 / every 3 layers | 13 | 187.45 ms | 110.61 ms | 1.695x | 0.999862 | 1.696% | 0.7824 | 72.23% |
| radius 2 / every 5 layers | 8 | 187.69 ms | 107.76 ms | 1.742x | 0.999845 | 1.914% | 0.8987 | 47.22% |

The first row was captured before boundary accounting was moved ahead of the
independent full-budget cache check, so its JSON contains a reset counter of
zero; the configured 38-layer interval has eight boundaries. Commit `71b1a58`
fixes the metric, and the two sweep rows report 13 and eight boundaries
directly. The forward outputs and timings of the earlier row remain valid.

Radius matters more than update frequency: increasing radius from one to two
at the same five-layer interval raises early video cosine by 0.1268 while also
improving the measured speed from 1.675x to 1.742x in this paired run. Merely
changing the radius-one interval from five to three layers yields only a 0.0105
cosine gain. Radius two every five layers is therefore the current propagation
candidate for eight-DiT replay, not yet a final quality configuration.

For completeness, radius one every five layers at the real late timestep 249
measures 2.614x, action cosine 0.999889, and video cosine 0.5158. Every
propagation row independently passes exact full-budget video, action, and cache
checks.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/dynamic_budgets/
    20260830_aggressive_tl_prop5_r1_gpu01/
    20260830_aggressive_tl_prop_sweep_early_gpu01/
```

## Interpretation and next gate

The timestep hypothesis is useful for compute allocation: the Oracle-supported
late budget produces a substantially smaller packed shape and a larger speedup.
The layer hypothesis must remain non-monotonic because late layers recover
strongly in the Dense Oracle.

However, one shared token budget across every head is not a viable final
policy. Raising all heads together spends nearly Dense compute on early DiT
without recovering video. Historical-KV-only head grouping is also not the
answer: it keeps current-token starvation unchanged and introduces too many
small attention launches. Spatial propagation demonstrates that restoring
information to unselected current tokens is the higher-leverage direction.

The next gate is therefore:

1. retain radius-two/every-five segmented propagation as the current candidate;
2. test a quality/aggressive hybrid current-token schedule, especially the
   Oracle-sensitive late-layer recovery region;
3. map calibrated per-request M1 confidence to a small number of shared groups
   or a single promoted group shape, avoiding 3--4 FA2 launches in every layer;
4. keep critical and confidence-uncertain routes Dense and log every fallback;
5. replay all eight real DiT evaluations on held-out DROID requests before any
   final quality claim;
6. keep VV extrapolation as a separately timed late-step module so its gain is
   never attributed to sparse attention.

The aggressive and quality tables remain performance/structure ablations. No
current dynamic table is promoted as the final M1 policy.
