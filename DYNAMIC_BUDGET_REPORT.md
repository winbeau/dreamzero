# Dynamic timestep/layer/head budget report

Date: 2026-08-31

## Status

The current shared timestep/layer table and request-wide fallback route is
closed as a negative result. The 108-request static table is consistently
faster but reaches only `1.0738x--1.0908x` by split and is action-safe on
92/108 requests. Maximum-Head promotion collapses to Dense, while the causal
request gate reaches only `1.009x--1.014x`. It does not meet the quality-safe
`>=1.35x` goal. See `FINAL_STATUS.md`.

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
inside the packed stack is substantially more effective. The follow-up sweep
also identifies a required executor invariant: mutable current-token budgets
must stay fixed within a propagation segment. Segment-max budgets reach 0.9749
video cosine at 1.552x early-DiT speedup, while a 75% segment floor reaches
0.9970 at 1.378x. The confidence-controlled M1 router now exists, but its
online history-feature connection and a full eight-DiT policy replay are still
required before this phase can pass its quality gate. The Packed-proxy v2
classifier is now connected to the two-rank executor and does produce causal
per-request Head groups, but its first validation pilot is a negative result:
the Q/K-coupled four-group realization is slower than Dense and fails the
action gate. It is retained as the required `timestep + layer + head`
ablation, not promoted as the main path.

Implementation commits:

- `28bbf47`: fixed-shape 8-by-40 dynamic budget table and Packed M2 runtime;
- `6fe77ec`: reproducible Oracle-ordered fixed/timestep/layer/joint ablations;
- `aa9706f`: per-rank early/late checkpoint gate and actual token accounting;
- `e7d8c2d`: align the checkpoint gate with the eight real diffusion timesteps;
- `dff03a7`: first fixed-membership dynamic packed head-group executor;
- `0aa67eb`: group heads per `(timestep, layer)` by shared M1 budget bucket;
- `45db7d4`: segmented packed spatial propagation without per-layer full scatter;
- `71b1a58`: reproducible per-rank propagation sweep and boundary accounting;
- `870be76`: compare per-rank dynamic budget tables in one checkpoint load;
- `4610143`: build propagation-boundary current-token sentinel ablations;
- `d839b08`: stabilize mutable current budgets within propagation segments;
- `39f0515`: localize conservative current compute by propagation segment;
- `ca0dce6`: enforce segment-stable current budgets inside Packed M2;
- `72a4c4d`: build fixed timestep-aware packed-segment policies.
- `22a8a3d`: risk-controlled calibrated M1 routing into four Q/K-coupled Head
  groups with downstream unknown/unsafe Dense fallback.
- `fa1d945`: causal online M1 feature state with first-two-DiT, missing-history,
  and observer-schema Dense fallback.
- `d163fff`: connect the Packed-proxy classifier, raw state conditions, causal
  observations, and live Head-group decisions to each of eight real DiTs.
- `194220e`: aggregate conditional and unconditional proxy signatures across
  the two inference-parallel ranks and fail Dense on non-finite required
  online signals.
- `390ac95`: bound prepacked QKV/O weight slices to the current membership
  partition instead of retaining a new full set for every DiT.

All listed commits are pushed to
`origin/codex/dreamzero-anchor-sparse-opt`, and the H200 checkout is
fast-forwarded through `72a4c4d`.
Commit `22a8a3d` is pushed but has not yet been synchronized to H200 because
both configured SSH routes were unavailable.

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

### Calibrated risk-controlled grouped route

Commit `22a8a3d` addresses the policy gap in the negative prior-only ablation
without reinterpreting its timing as a positive result. The new deployment
router consumes the frozen v3 classifier, calibrated confidence, and promotion
policy per request. It rounds each Head budget upward into `[25, 50, 75,
100]%`, couples current Q/K and historical K/V to the same ratio, and therefore
produces no more than four execution shapes per timestep/layer. This is the
requested Q compression path; the 25 action/state tokens remain Dense.

Four fallback causes remain distinct in runtime diagnostics:

- classifier confidence below its frozen threshold;
- missing or mismatched online feature-observer provenance;
- downstream group evidence that violates explicit action/video thresholds;
- no task-disjoint downstream evidence for that Head cell.

All three force the affected Head to 100%. The downstream table accepts only
scale-zero group-removal records with exact trace agreement, the configured
split and trajectory-stage coverage, and a minimum unique-request count. This
prevents the incomplete current scan from being silently treated as safety
evidence.

The companion evaluator reruns validation and test without retuning, rounds
Oracle truth upward to the same executor buckets, and reports post-grouping
false-sparse rate, macro-F1, mass retention, calibration, group count,
fallback composition, and 200-repeat episode bootstrap. The causal state added
in `fa1d945` preserves the exact eight-real-DiT order and refuses to use the
current v3 bundle sparsely because that artifact does not declare a compatible
online observation schema. Twenty-seven focused and existing online-state,
M1, and dynamic-budget tests pass together with Ruff, compilation, and diff
checks.

At that commit boundary there was deliberately no speed or quality row because
H200 was unreachable. The following section records the later online replay;
unknown downstream cells still route Dense in the default safe configuration,
and the coverage-disabled pilot is explicitly an ablation rather than an
accepted result.

### Packed-proxy v2 online runtime pilot

The server path now loads `selected_m1_bundle.joblib`, computes the raw
pre-normalization `state_l2` and `state_abs_mean` features used at training,
and routes immediately before each actual DiT call. Scheduler skips do not
advance M1. DiT 0 and 1 are forced Dense; DiT `t` can consume only observations
completed by DiTs `< t`. Missing state, observer schema mismatch, incomplete
CFG aggregation, non-finite required one-step history, low confidence, and
unknown/unsafe downstream evidence each remain explicit Dense fallbacks.

The first two-rank pilot exposed two deployment bugs before a sparse claim was
allowed:

- each rank saw only one CFG branch, so the single-rank observer produced zero
  valid observations and every step correctly fell back Dense;
- after CFG aggregation was added, request-dependent Head membership retained
  a new prepacked QKV/O weight set at each DiT, growing by roughly 8.4 GiB per
  dynamic step and causing an H200 OOM on the following request.

Commits `194220e` and `390ac95` close those bugs. Fifty-one focused H200 tests
pass. In the corrected real service every request records 8/8 valid proxy
observations. The measured example routes DiTs 0--1 Dense, then uses the
following grouped budgets:

| real DiT | Mean keep | Dense Head fraction | Confidence fallback | Mean groups/layer |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 84.41% | 55.00% | 30.94% | 3.50 |
| 3 | 84.55% | 56.38% | 32.62% | 3.48 |
| 4 | 84.81% | 55.88% | 32.62% | 3.42 |
| 5 | 84.92% | 57.06% | 35.19% | 3.48 |
| 6 | 85.98% | 60.69% | 39.12% | 3.38 |
| 7 | 86.45% | 61.69% | 41.00% | 3.50 |

This pilot deliberately disabled the incomplete downstream-coverage gate to
measure the classifier policy; it is not a safe main result. On three paired
validation targets it gives:

| Policy | Dense mean | Dynamic mean | Paired geomean speedup | Faster fraction | Action cosine mean/min | Action rel-L2 mean/max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| proxy-v2 Q/K-coupled four groups | 2.325 s | 2.667 s | 0.841x | 33.3% | 0.98096 / 0.95802 | 17.00% / 29.16% |

The latency and quality gates both fail decisively. The result explains why:
3.4--3.5 varlen groups add projection/packing overhead, while 55--62% Dense
Heads force the shared packed activation and FFN close to full current length.
Sparse Heads still lose query outputs, so compute does not fall enough to pay
for the quality loss.

The bounded-cache stress repeats four target trajectories after the first
four without an OOM. GPU memory plateaus near 126.8/128.7 GiB instead of the
unbounded version's 139.8 GiB failure, and the second pass latency is stable at
2.620--2.627 s. This validates the cache fix, not the policy.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/runtime/20260831_dynamic_m1_d163fff_pilot/
    dense_validation4.json
    sparse_validation4_v3.json
    sparse_validation4_v3_repeat.json
    comparison_validation3_v3.json
    dense_server.log
    sparse_server_v3.log
```

The next executor revision will therefore separate the axes: Head class will
control historical K/V, while current Q/FFN uses a small layer-shared bucket
chosen conservatively from the routed Heads. Per-Head Q compression remains
as the negative ablation above.

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

## Propagation-aligned current-budget invariant

The initial dynamic table allowed current/Q budget to shrink and later expand
inside one five-layer packed segment. This is invalid for mutable hidden state:
a token removed at one layer skips that Transformer update, then re-enters a
later layer with a stale representation. Historical K/V budget may still vary
per layer because those gathered cache entries are immutable inputs; current
Q/O/FFN state may change only at a propagation/repack boundary.

A boundary-only sentinel first exposed the failure mode. Promoting more current
tokens only in the final layer of each segment made the output worse, because
the newly activated tokens had skipped all preceding layers in that segment:

| Boundary-only current sentinel | Early current mean | Dense p50 | Sparse p50 | Speedup | Action cosine | Video cosine | Video rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 75% | 53.42% | 187.92 ms | 112.13 ms | 1.676x | 0.999794 | 0.8344 | 60.37% |
| 100% | 58.68% | 187.16 ms | 118.64 ms | 1.578x | 0.999764 | 0.7133 | 86.61% |

The corrected table holds current budget constant at the maximum requested
ratio within each propagation segment. This prevents stale-token re-entry while
retaining timestep-dependent and segment-dependent fixed buckets. Historical
K/V remains the original aggressive timestep/layer table.

| Segment-stable current policy | Early current mean | Dense p50 | Sparse p50 | Speedup | Action cosine | Action rel-L2 | Video cosine | Video rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| segment maximum | 59.87% | 190.48 ms | 122.76 ms | 1.552x | 0.999910 | 1.346% | 0.9749 | 22.57% |
| segment floor 75% | 75.00% | 187.61 ms | 136.15 ms | 1.378x | 0.999820 | 1.914% | 0.9970 | 7.75% |

The invariant is now enforced inside M2 rather than relying only on generated
tables. Whenever packed propagation is active, the executor replaces current
budgets inside each segment by that segment's maximum and logs those effective
ratios. M1 may still predict per-layer history KV budgets, but it cannot
reactivate stale mutable current state. Commit `ca0dce6` passes 24 focused tests
and a real 14B checkpoint path.

This is a much larger quality recovery than making only the Dense suffix
deeper. With the same radius-two/every-five propagation, four Dense suffix
layers reach 0.9058 video cosine at 1.687x, while eight Dense suffix layers
reach 0.9179 at 1.535x. The remaining error is accumulated across packed
segments, not confined to the output layers.

All rows use the real early timestep 999 and do not change the service-level
contract of eight actual DiT evaluations. Each row independently passes exact
full-budget video, action, and cache checks. These remain single-timestep
released-checkpoint gates rather than held-out task or closed-loop claims.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/dynamic_budgets/
    20260830_aggressive_tl_prop_r2_suffix_sweep_early_gpu01/
    20260830_propagation_sentinels/
    20260830_prop_r2_sentinel75_100_early_gpu01/
    20260830_prop_r2_segmentmax_floor75_early_gpu01/
```

## Early-segment localization

Equal-compute gates then localized the four early propagation segments that
the 75% floor changes at DiT 0. The last four segments already use 75% in the
Oracle-ordered early table.

| Current segments promoted to 75% | Layers | Sparse p50 | Speedup | Action cosine | Video cosine | Video rel-L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| segment 0 | 1--5 | 122.77 ms | 1.529x | 0.999901 | 0.9841 | 17.81% |
| segment 1 | 6--10 | 122.84 ms | 1.525x | 0.999901 | 0.9820 | 19.06% |
| segment 2 | 11--15 | 127.76 ms | 1.490x | 0.999879 | 0.9818 | 19.15% |
| segment 3 | 16--20 | 122.76 ms | 1.528x | 0.999894 | 0.9728 | 23.48% |
| segments 0 + 1 | 1--10 | 126.58 ms | 1.482x | 0.999880 | 0.9887 | 15.03% |
| segments 2 + 3 | 11--20 | 128.33 ms | 1.457x | 0.999859 | 0.9805 | 19.81% |
| segments 0 + 1 + 2 | 1--15 | 132.60 ms | 1.417x | 0.999832 | 0.9962 | 8.76% |
| segments 0 + 1 + 3 | 1--10, 16--20 | 130.06 ms | 1.443x | 0.999861 | 0.9894 | 14.55% |

The first 15 layers are the best early allocation: they recover nearly all of
the full 75% floor's video cosine (0.9962 versus 0.9970) at a higher speedup
(1.417x versus 1.378x). Segment 3 is weak alone and inferior to segment 2 in
the equal-size composite. This supports the proposed story that early visual
anchors and their first representation refinements are disproportionately
important; it does not support a generic monotonic layer-depth rule.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/dynamic_budgets/
    20260830_prop_r2_segment_groups_early_gpu01/
    20260830_prop_r2_segment0_vs1_early_gpu01/
    20260830_prop_r2_segment2_vs3_early_gpu01/
    20260830_prop_r2_segment012_vs013_early_gpu01/
```

## Timestep-aware packed-segment policies

Two fixed eight-DiT profiles combine the supported timestep ordering with the
segment evidence. Both keep the exact scheduler/DiT contract and use the same
nested route ordering:

- `timestep_segment_balanced`: effective current means 71.71% early, 61.84%
  middle, and 36.97% late;
- `timestep_segment_quality`: the same early policy, 65.13% middle, and 50.00%
  late.

The historical-KV row remains the aggressive Oracle-ordered table and falls
with timestep independently of current Q/O/FFN compute.

| Policy | DiT / timestep | History mean | Current mean | Sparse p50 | Speedup | Action cosine | Video cosine | Video rel-L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | middle / 892 | 39.87% | 61.84% | 119.20 ms | 1.577x | 0.999795 | 0.9834 | 18.18% |
| balanced | final / 249 | 20.79% | 36.97% | 96.60 ms | 1.966x | 0.999767 | 0.8830 | 50.84% |
| quality | late / 535 | 28.29% | 50.00% | 107.09 ms | 1.775x | 0.999789 | 0.9649 | 26.76% |
| quality | final / 249 | 20.79% | 50.00% | 101.26 ms | 1.853x | 0.999659 | 0.9228 | 40.61% |

The middle balanced gate is promising. Both late gates retain the local action
threshold but have substantial isolated video error, so neither profile is
promoted solely from these rows. Late-step state is produced by the preceding
denoise trajectory, and the required next decision must therefore come from a
paired full eight-DiT replay. The aggressive 36.97% late row remains a speed
ablation until that replay passes final video/action quality.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/dynamic_budgets/
    20260830_timestep_balanced_middle_late_gpu01/
    20260830_timestep_quality_late_gpu01/
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
The segment-stability sweep further shows that arbitrary layer-by-layer Q
budgets are mathematically unsafe even when their shapes are nested.

The next gate is therefore:

1. enforce segment-stable current budgets in M2 and retain radius-two/every-five
   propagation as the current candidate;
2. replay the balanced and, if needed, quality policy through all eight real
   DiT evaluations on paired held-out requests before selecting late budgets;
3. map calibrated per-request M1 confidence to a small number of shared groups
   or a single promoted group shape, avoiding 3--4 FA2 launches in every layer;
4. keep critical and confidence-uncertain routes Dense and log every fallback;
5. replay all eight real DiT evaluations on held-out DROID requests before any
   final quality claim;
6. keep VV extrapolation as a separately timed late-step module so its gain is
   never attributed to sparse attention.

The aggressive and quality tables remain performance/structure ablations. No
current dynamic table is promoted as the final M1 policy.

## Full eight-DiT balanced-policy smoke

The balanced table was then replayed through the real WebSocket policy path on
three paired measured requests plus one warmup. Dense used physical GPUs 2--3
and Sparse used 5--6. Both services retained all 16 scheduler steps and every
request log reports exactly eight real DiT evaluations.

| Metric | Dense | Balanced Sparse | Result |
| --- | ---: | ---: | ---: |
| client end-to-end mean | 2.0616 s | 1.5078 s | 1.367x |
| server inference mean | 2.0507 s | 1.4940 s | 1.373x |
| diffusion mean | 1.4800 s | 1.0767 s | 1.375x |
| measured requests faster | -- | 3 / 3 | 100% |
| action cosine | -- | mean 0.999634, min 0.999485 | pass |
| action relative L2 | -- | mean 2.696%, max 3.292% | pass |

This is the first full-trajectory evidence that the balanced late budget does
not accumulate enough error to fail the action gate. It is still only a smoke
test: three requests on one GPU assignment cannot establish the required
confidence interval, task coverage, video quality, or closed-loop
non-inferiority. The balanced table is promoted to the candidate for the
larger paired run, not to the final policy.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/e2e/20260830_balanced_smoke/
    dense.json
    sparse.json
    comparison.json
    dense_log_summary.json
    sparse_log_summary.json
```

## Real DROID history-chain rejection

The random-image smoke above was not sufficient to select a policy. The same
service path was therefore driven by the task-disjoint DROID Oracle manifest:
one real episode, its early/middle/late instructions and states, and three
historical four-frame blocks before each measured target. Every history and
target call retained eight real DiT evaluations.

The full-budget Packed path exactly matches Dense after the complete history
chain (`action cosine = 1.0`, relative L2 `= 0`), so reset, distributed state,
and Packed M2 exactness are not the source of the following errors.

| Policy | History mean | Current mean | Target mean speedup | Middle cosine / rel-L2 | Late cosine / rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced | 37.50% | 56.19% | 1.421x | 0.990942 / 13.43% | 0.988943 / 16.10% |
| quality | 37.50% | 61.23% | 1.419x | 0.996679 / 8.25% | 0.981261 / 20.37% |
| quality + history floor 75% | 75.00% | 61.23% | 1.157x | 0.993865 / 11.24% | 0.990773 / 13.95% |
| quality + Dense history | 100.00% | 61.23% | 1.264x | 0.998383 / 5.77% | 0.997600 / 8.17% |
| current floor 75% + Dense history | 100.00% | 74.44% | 1.183x | 0.999405 / 4.05% | 0.997019 / 8.06% |
| full-budget Packed | 100.00% | 100.00% | 0.982x | 1.000000 / 0% | 1.000000 / 0% |

The latency rows contain only two measured target requests and are diagnostic,
not paper estimates. The quality result is nevertheless decisive: no shared
global table in this sweep meets the required worst-request action threshold.
Even the conservative 75% current / Dense history table passes the middle
target but fails the late target. The balanced table is therefore rejected as
the candidate for the 100-request run.

The next policy must use calibrated request/head confidence to promote or
fallback before executing the target. A low-confidence trajectory cannot be
assigned a universal 75% fallback; the demonstrated late request requires a
Dense fallback. Historical-K/V budget must also be part of the confidence
decision rather than remaining the aggressive Oracle average.

Artifacts:

`dynamic_m1_m2/e2e/20260830_droid_108_round1/`

## Expanded 108-request global-profile gate

The global profiles were replayed over 72 train, 18 validation, and 18
untouched test requests. Every target has three real history calls and every
history/target call executes eight real DiT evaluations.

| Profile / split | Requests | Mean speedup | Quality-safe | Cosine mean / min | Relative-L2 mean / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced / train | 72 | 1.453x | 8 / 72 | 0.99455 / 0.97130 | 9.95% / 25.01% |
| balanced / validation | 18 | 1.476x | 0 / 18 | 0.99554 / 0.98323 | 9.29% / 18.35% |
| balanced / test | 18 | 1.493x | 2 / 18 | 0.99332 / 0.97178 | 10.60% / 23.60% |
| 75% current + Dense history / train | 72 | 1.191x | 36 / 72 | 0.99793 / 0.98578 | 5.84% / 17.11% |
| 75% current + Dense history / validation | 18 | 1.194x | 7 / 18 | 0.99660 / 0.97059 | 6.71% / 25.35% |
| 75% current + Dense history / test | 18 | 1.173x | 11 / 18 | 0.99675 / 0.95834 | 5.73% / 28.59% |

All sparse requests are faster, but neither global sparse table is a
quality-safe policy. An Oracle request selector over `balanced`,
`conservative`, and `dense` reaches only 1.1085x/1.0683x/1.1254x mixed
end-to-end speedup on train/validation/test. The current global-profile family
has a hard performance ceiling below target even with perfect classification.

## First-two-DiT Dense ablation

Making DiT indices 0 and 1 Dense and applying the balanced table to the last
six real evaluations tests whether only the earliest denoising steps require
high budget.

| Split | Speedup | Quality-safe | Cosine mean / min | Relative-L2 mean / max |
| --- | ---: | ---: | ---: | ---: |
| validation | 1.373x | 1 / 18 | 0.99607 / 0.98526 | 9.01% / 17.11% |
| test | 1.425x | 3 / 18 | 0.99414 / 0.96760 | 9.77% / 25.39% |

This rejects a static “first two Dense, remaining six sparse” mask. Early
budget is important but insufficient; later packed-trajectory error must be
controlled by finer layer/head routing or propagation, not only a timestep
prefix.

Artifacts:

```text
dynamic_m1_m2/e2e/20260830_droid_108_round1/
  request_gate_train72/
  request_gate_validation18/
  request_gate_balanced_val_test/
  early2_dense_then_balanced_val_test/
```

## Two-group current-QKV executor gate

The deployment-safe v3 M1 prior was quantized into critical and normal head
groups with separate historical-KV and current-QKV prefixes. The executor
slices video Q/K/V/O channels, keeps all 25 action/state registers Dense,
prepackages fused QKV/O weights, and combines heterogeneous head sequences in
one FA2 varlen launch. It passes 29 focused tests, and the released 14B
full-budget video/action/all-layer-KV path remains exactly equal to Dense.

The systems result is negative. At the released geometry with 12 critical and
28 normal heads, one regular 40-head call is 1.54 ms, the old two-group path is
2.20 ms, temporary sliced projections are 2.36 ms, prepacked two-call
projections are 2.31 ms, and the final one-launch head-as-batch varlen path is
2.11 ms. Reducing the critical set to 4 or 8 heads does not help: varlen p50 is
2.13 and 2.33 ms versus 1.48 and 1.53 ms for regular FA2.

| Outer trunk / head budgets | DiT speedup | Action cosine / rel-L2 | Video cosine / rel-L2 |
| --- | ---: | ---: | ---: |
| 50%; critical H100/Q50, normal H35/Q25 | 1.063x p50 | 0.999908 / 1.451% | 0.9532 / 32.26% |
| 35%; critical H100/Q35, normal H25/Q20 | 1.317x p50 | 0.999901 / 1.543% | 0.8783 / 54.51% |

The 50% samples continue from 176.45 to 135.73 ms after the measured p50; even
the last sample is only 1.38x against the paired Dense median. Per-head varlen
is therefore retained as an ablation/fallback, not the main kernel. Main-path
M1/M2 must merge heads to shared fixed shapes whenever a measured cost model
predicts that heterogeneous execution loses throughput.

Implementation commits: `24bff11`, `b9dd2f6`, `6e68569`, `f6c3ff4`,
`ee775fb`, and `42330a8`.

Artifacts:

```text
dynamic_m1_m2/packed_m2/20260831_head_sliced_microbench/
dynamic_m1_m2/dynamic_budgets/20260831_two_group_qkv/
  checkpoint_early_gpu01/
  checkpoint_early_gpu01_varlen_lowtrunk/
```

## Dense action-history route ablation

The packed executor now supports a protected action route: sparse video
queries use the selected historical prefix, while the 25 action/state queries
attend the complete historical K/V. This directly tests whether accumulated
action error is dominated by sparse action-to-history attention without
restoring Dense video Q/K/V/O or FFN.

The paired checkpoint result improves action relative L2 from 1.849% to
1.703%, but costs 9.45 ms Sparse p50. On the complete validation18 history
chains, mean relative L2 improves by 9.85% relative (9.293% to 8.377%) and
12/18 requests improve individually. The mechanism is not monotonic: six
requests regress, the worst late request reaches 21.06% relative L2, and the
quality-safe count remains 0/18. Mean end-to-end speedup is 1.332x with paired
CI95 [1.306x, 1.354x].

This isolates an important design constraint for dynamic M1: action-history
protection may be selected for particular high-confidence routes, but it is
not itself a safe fallback. Low-confidence and demonstrated regression cases
still require budget promotion or the exact Dense path.

Artifacts:

```text
dynamic_m1_m2/dynamic_budgets/20260831_dense_action_history/
dynamic_m1_m2/e2e/20260831_dense_action_history_balanced_validation18/
```

## DiT/layer action-history scheduling

Commit `69a32c6` adds a strict eight-DiT by forty-layer boolean schedule for
the protected action-history call. This makes the mechanism compatible with
the paper's `r(timestep, layer, head_class)` structure without introducing
arbitrary token shapes: each cell selects either the existing sparse-history
call or the existing 25-query Dense-history call.

The first layer-bucket gate compares the same balanced current/history budget
at real DiT index zero:

| Protected layers | Sparse p50 | DiT speedup | Action cosine | Action rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| 1--13 | 154.57 ms | 1.217x | 0.999920 | 1.515% |
| 28--38 | 157.91 ms | 1.202x | 0.999842 | 1.837% |
| all 1--38 | 164.43 ms | 1.147x | 0.999863 | 1.703% |
| none | 154.98 ms | 1.228x | 0.999832 | 1.849% |

Thus complete action history is not uniformly beneficial across layer depth;
the early bucket dominates the late and all-layer variants on this input. The
held-out validation18 replay reaches 1.402x mean speedup but only reduces mean
relative L2 from 9.293% to 8.972%, with zero quality-safe requests. The layer
effect is real enough to retain as an M1 feature, but the fixed early-layer
schedule is rejected as a policy.

At real DiT index 4 (timestep 749), the same early-layer protection remains
useful rather than changing sign: action relative L2 improves from 2.487% to
1.457% and cosine from 0.999784 to 0.999902. The price is 132.31 to 145.59 ms
Sparse p50, reducing DiT speedup from 1.417x to 1.308x. Therefore action-history
protection is both layer- and timestep-sensitive, but restricting it to only
early denoising steps would discard a large late-step quality gain. Since the
all-eight-DiT early-layer service still fails 18/18 validation requests, the
next budget experiment must protect the video-query history path that creates
future cache state, not only the 25 action queries.

Artifacts:

```text
dynamic_m1_m2/dynamic_budgets/20260831_dynamic_action_history/
  checkpoint_early_vs_late_gpu01/
  checkpoint_dit4_early_vs_none_gpu01/
dynamic_m1_m2/e2e/20260831_dynamic_action_history_early_layers_validation18/
```

## Early-layer video-history floor

Commit `55c1051` builds reproducible variants that raise only layers 1--13 of
the balanced video-query historical K/V table to 50%, 75%, or 100%, while
leaving every current-token budget unchanged. At DiT index zero, the 75% floor
outperforms the 100% floor: action relative L2 is 1.259% versus 1.360%, video
relative L2 is 8.525% versus 8.590%, and Sparse p50 is 148.83 versus 148.01 ms.
This is further evidence that more historical mass is not monotonically safer.

The validation18 replay rejects the 75% floor as a global policy. It reaches
1.487x ratio-of-means speedup and 18/18 Sparse-faster requests, but mean action
relative L2 worsens from 9.293% to 10.011% and worst L2 reaches 23.41%. Two
requests pass the action gate, including one request not covered by the prior
conservative profile. Across balanced, conservative, both action-history
variants, and early-video-history75, the sparse-profile safety union is only
8/18. Even a perfect per-request Oracle reaches just 1.102x ratio-of-means
speedup because ten requests still require Dense execution. Coarse global
history floors are therefore exhausted as a route family.

Artifacts:

```text
dynamic_m1_m2/dynamic_budgets/20260831_early_video_history/
dynamic_m1_m2/e2e/20260831_early_video_history75_validation18/
```

## Dense-suffix recovery gate

The balanced packed table was next evaluated with three and five Dense suffix
layers, keeping the Dense prefix, dynamic budgets, radius-two/every-five
propagation, and all eight real DiT calls fixed.  On the released early-step
checkpoint, increasing the suffix from one to three to five layers reduces
video relative L2 from 8.758% to 7.512% to 6.708%.  Action relative L2 remains
nearly flat at 1.849%, 1.858%, and 1.864%, while paired DiT speedup changes
from 1.228x to 1.252x to 1.159x.  Full-budget video, action, and cache paths are
exact for both new rows.

The complete validation18 replay shows that the checkpoint video recovery does
not translate into monotonic final-action recovery:

| Dense suffix | Paired geomean speedup | CI95 | Action cosine mean/min | Action rel-L2 mean/max | Safe |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 1.476x | [1.459x, 1.493x] | 0.995541 / 0.983234 | 9.293% / 18.352% | 0/18 |
| 3 | 1.484x | [1.431x, 1.518x] | 0.995709 / 0.978420 | 8.999% / 20.828% | 1/18 |
| 5 | 1.420x | [1.377x, 1.457x] | 0.995497 / 0.976171 | 9.103% / 21.758% | 0/18 |

Every suffix-3 and suffix-5 target is faster than Dense and every one of the
76 warmup/history/target calls per service records exactly eight DiT model
evaluations.  Suffix three gives only a small mean improvement and worsens the
tail; suffix five loses both speed and action quality.  Dense output recovery
is therefore saturated and non-monotonic.  The dominant error is accumulated
inside packed propagation segments, so further recovery work must control
segment state rather than merely extending the output suffix.

Artifacts:

```text
dynamic_m1_m2/dynamic_budgets/20260831_dense_suffix_recovery/
dynamic_m1_m2/e2e/20260831_balanced_suffix3_validation18/
dynamic_m1_m2/e2e/20260831_balanced_suffix5_validation18/
```

## Propagation frequency and spatial-radius recovery

After rejecting deeper suffix recovery, the same balanced suffix-one executor
was tested with a shorter propagation segment and a wider spatial update.  The
released early-step checkpoint gives:

| Propagation | Boundaries | Sparse p50 | DiT speedup | Action rel-L2 | Video cosine | Video rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| radius 2 / every 5 | 8 | 154.98 ms | 1.228x | 1.849% | 0.996166 | 8.758% |
| radius 2 / every 3 | 13 | 153.07 ms | 1.226x | 1.863% | 0.996098 | 8.830% |
| radius 3 / every 5 | 8 | 147.14 ms | 1.272x | 1.839% | 0.999155 | 4.233% |

Frequency alone is ineffective: thirteen radius-two recoveries are slightly
worse than eight.  A radius-three update, however, cuts the local video error
by 51.7% without changing the packed token budgets.  This result independently
passes full-budget video, action, and cache exactness and was promoted to the
complete validation replay.

The trajectory result again separates video reconstruction from action-state
safety.  Radius three/every five measures 1.2563 s mean target latency versus
1.9034 s Dense, with 1.516x paired geometric mean speedup and CI95 [1.485x,
1.542x].  All 18 targets are faster and all 76 warmup/history/target calls
execute exactly eight DiT evaluations.  Nevertheless mean action relative L2
regresses to 9.553%, minimum cosine is 0.979434, maximum L2 is 20.289%, and no
request passes both action gates.

The spatial-radius sweep is therefore closed as a global recovery mechanism.
Wider propagation can repair the observable video field while failing to
repair the action-sensitive packed register/hidden-state trajectory.  The next
executor change must target action-conditioned state inside a segment rather
than continue increasing propagation radius or frequency.

Artifacts:

```text
dynamic_m1_m2/dynamic_budgets/20260831_propagation_recovery/
dynamic_m1_m2/e2e/20260831_balanced_r3e5_validation18/
```

## Dynamic maximum-current action readout

Commit `bd28485` adds an optional fixed 8x40 schedule for a narrower
action-sensitive recovery experiment. At an enabled cell, only the 25
action/state queries read K/V from the maximum already-packed current-video
prefix. Video queries, current Q/O, cross-attention, and FFN retain the active
layer prefix, so this does not scatter the middle state or restore full-token
compute. The schedule generator separates propagation-segment entries
`{1,6,11,16,21,26,31,36}` from exits
`{5,10,15,20,25,30,35,38}` for radius-two/every-five Packed M2.

Two H200 checkpoint rounds exchanged the entry and exit candidates across
GPUs 0 and 1. A same-commit two-GPU `none` run reproduced the original quality
exactly:

| Maximum-current cells | Action cosine | Action rel-L2 | Video cosine | Video rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| none | 0.99983209 | 1.84871% | 0.99616599 | 8.75774% |
| segment entries | 0.99983412 | 1.84485% | 0.99617040 | 8.75309% |
| segment exits | 0.99982333 | 1.93281% | 0.99616808 | 8.75578% |
| all packed layers | 0.99978894 | 2.14739% | 0.99616653 | 8.75699% |

The same schedule produces bit-identical quality metrics after the GPU swap.
Fresh segment-entry readout improves action relative L2 by only 0.00387
percentage points, or 0.21% relative. Reading the under-updated maximum prefix
at segment exits worsens action L2 by 4.55% relative, and enabling it in all
packed layers worsens L2 by 16.16%. This directly verifies that dormant
maximum-prefix tokens become stale inside a propagation segment.

The entry and exit candidates have essentially identical exchanged-round
geometric-mean checkpoint speedups, 1.26235x and 1.26225x. Independent `none`
timing is too noisy to infer a sub-millisecond overhead: each rank has a large
first-use compilation sample and substantial run-to-run drift. Since the only
reproducible quality gain is 0.21% relative and the accumulated validation
failure is orders of magnitude larger, maximum-current action readout is
retained as a negative ablation and is not promoted to validation18 or the
dynamic policy.

Artifacts:

```text
dynamic_m1_m2/dynamic_budgets/20260831_max_action_current/
  checkpoint_balanced_gpu01/
  checkpoint_segment_entries_vs_exits_gpu01_bd28485/
  checkpoint_segment_exits_vs_entries_swap_gpu01_bd28485/
  checkpoint_none_both_gpu01_bd28485/
  tables/
```

## Runtime axis-separation pilot

The first online M1 replay coupled each Head's current Q shape to its
historical K/V budget.  Commits `640f001` and `117d08d` added a layer-shared
current-token bucket so Head class can control only historical K/V.  The
following task-aligned pilot uses the same three measured validation requests,
the same Dense artifact, 16 scheduler steps, and eight real DiT evaluations per
inference.  These rows are diagnostic because downstream Head-risk coverage is
still incomplete and the online M1 rows deliberately disable that gate.

| Policy | Paired E2E speedup | Faster | Action cosine mean/min | Action rel-L2 mean/max |
| --- | ---: | ---: | ---: | ---: |
| M1 Head Q/K coupled | 0.841x | 1/3 | 0.98096 / 0.95802 | 17.00% / 29.16% |
| M1 Head K + shared Q75 | 0.929x | 1/3 | 0.98251 / 0.95277 | 15.53% / 32.82% |
| fixed K100 + Q75, radius3/every5 | 1.351x | 3/3 | 0.98287 / 0.95152 | 15.28% / 33.55% |
| fixed K75 + Q100 | 1.235x | 3/3 | 0.99865 / 0.99712 | 5.41% / 9.30% |
| M1 Head K + Q100 | 0.784x | 1/3 | 0.99746 / 0.99365 | 6.89% / 11.63% |

The axis separation changes the diagnosis.  Keeping every historical K/V token
does not rescue Q75: the worst request still reaches 33.55% action relative
L2.  Keeping current Q/K/V/O and FFN Dense while uniformly reducing history to
75% is much safer, but remains outside the 0.999 cosine and 5% relative-L2
gates.  Therefore skipped current-token hidden-state updates are the dominant
error source, while historical K sparsity is a smaller but still measurable
source.

The unguarded M1 Head-K policy is worse than the shared K75 policy and is also
slower than Dense.  Its 3--4 Head groups add projection and varlen launch
overhead, and the proxy classifier's missing downstream coverage allows a few
action-critical Heads to receive sparse history.  Per-Head execution is now a
negative ablation, not the main acceleration path.  The next runtime schedule
will use a small number of timestep/layer shared shapes and require calibrated
downstream coverage or Dense fallback before any Head-specific reduction.

Artifacts:

```text
dynamic_m1_m2/runtime/20260831_dynamic_m1_d163fff_pilot/
  sparse_sharedq75_validation4.json
  comparison_sharedq75_validation3.json
  sparse_k100_q75_p3e5_a4427db_droid_validation4.json
  comparison_k100_q75_p3e5_a4427db_droid_validation3.json
  sparse_k75_q100_a4427db_droid_validation4.json
  comparison_k75_q100_a4427db_droid_validation3.json
  sparse_dynamic_headk_q100_a4427db_droid_validation4.json
  comparison_dynamic_headk_q100_a4427db_droid_validation3.json
```

## Oracle-guarded shared timestep/layer pilot

Commit `f9429d2` adds a deployment-safe table builder that ranks Dense-Oracle
timestep/layer cells, preserves an explicit Dense DiT prefix, and emits only
shared shapes.  The first table selects DiT indices 5--7 and 20 low-risk
layers, or 60/320 (18.75%) of all DiT/layer cells.  It therefore avoids the
3--4 per-Head kernel groups that made the unguarded online-M1 replay slower
than Dense.

The following real-DROID pilot reuses the same three measured task keys and
Dense baseline as the axis-separation study.  Every row executes all eight
real DiTs inside the fixed 16-step scheduler.

| Guarded shared policy | Paired E2E geometric speedup | 95% CI lower | Faster | Action cosine mean/min | Action rel-L2 mean/max |
| --- | ---: | ---: | ---: | ---: | ---: |
| history K75, current Q100 | 1.223x | 1.003x | 3/3 | 0.999775 / 0.999447 | 1.90% / 3.34% |
| history K75, current Q75, radius3/every5 | 1.222x | 1.003x | 3/3 | 0.999932 / 0.999907 | 1.17% / 1.38% |
| history K75, current Q50, radius3/every5 | 1.241x | 1.029x | 3/3 | 0.999754 / 0.999483 | 2.14% / 3.40% |
| history K75, current Q35, radius3/every5 | 1.216x | 1.009x | 3/3 | 0.999536 / 0.999400 | 3.06% / 3.63% |

All four rows pass the preliminary action cosine and relative-L2 gates.  Q35
is the most aggressive quality-safe bucket tested so far.  Its lower timing
than Q50 is not evidence that extra arithmetic is faster: with the current
propagation contract, only complete five-layer sparse segments retain the
requested current-token shape.  Non-contiguous selected cells inside a mixed
segment are promoted to the segment maximum, while their historical K/V can
still use K75.  The current table therefore exposes only a small amount of
Q35 work, and three requests are far too few to resolve the remaining launch
and system noise.

This result validates the shared timestep/layer fallback direction but does
not validate the final dynamic policy.  The next builder revision must score
and select propagation-aligned contiguous layer segments so that the Oracle
budget equals the shape actually executed by Packed M2.  Q25 is also retained
as a boundary ablation.  Neither result will be promoted without task-disjoint
coverage, at least 100 paired requests, GPU exchange rounds, video gates, and
closed-loop evaluation.

Artifacts:

```text
dynamic_m1_m2/dynamic_budgets/
  20260831_guarded_shared_late3_safe20/
  20260831_guarded_shared_late3_safe20_h75q50/
  20260831_guarded_shared_late3_safe20_h75q35/
dynamic_m1_m2/runtime/20260831_dynamic_m1_d163fff_pilot/
  comparison_guarded_history_late3_safe20_f9429d2_droid_validation3.json
  comparison_guarded_joint_late3_safe20_r3e5_f9429d2_droid_validation3.json
  comparison_guarded_joint_late3_safe20_h75q50_r3e5_f9429d2_droid_validation3.json
  comparison_guarded_joint_late3_safe20_h75q35_r3e5_f9429d2_droid_validation3.json
```

## Propagation-aligned Oracle frontier

Commit `5eb04fb` extends the guarded table builder to rank complete Packed-M2
propagation segments.  The default segment risk is the maximum Dense-Oracle
layer score inside the segment, so one unusually sensitive layer prevents an
otherwise low-mean segment from being selected early.  With prefix/suffix one
and propagation every five layers, the first four ranked segments are layers
1--20.  Unlike the earlier individual-layer table, every requested current
budget in these segments is the budget actually executed.

The same three measured real-DROID task keys give the following boundary
pilot.  Rows labelled repeat are taken after the service has already compiled
all shapes; every row still executes eight real DiTs.

| Schedule | Sparse cells | Mean latency | Paired geomean speedup | Faster | Action cosine mean/min | Action rel-L2 mean/max | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| late3, 4 segments, H75/Q50 | 60/320 | 1.791 s | 1.252x | 3/3 | 0.999695 / 0.999317 | 2.28% / 3.87% | pass |
| late3, 4 segments, H75/Q35 | 60/320 | 1.810 s | 1.239x | 3/3 | 0.998594 / 0.996235 | 4.27% / 8.72% | reject |
| late3, 4 segments, H50/Q50, repeat | 60/320 | 1.768 s | 1.269x | 3/3 | 0.999700 / 0.999375 | 2.44% / 3.93% | pass |
| late4, 4 segments, H50/Q50, repeat | 80/320 | 1.708 s | 1.313x | 3/3 | 0.999569 / 0.999218 | 2.92% / 4.09% | pass |
| late5, 4 segments, H50/Q50 | 100/320 | 1.693 s | 1.325x | 3/3 | 0.994347 / 0.984626 | 8.86% / 18.41% | reject |
| late4, 5 segments, H50/Q50 | 100/320 | 1.770 s | 1.267x | 3/3 | 0.999158 / 0.998245 | 3.78% / 5.92% | reject |

The safe pilot frontier is therefore `late4 x four segments x H50/Q50`.
Moving the same 100 sparse cells onto an earlier denoise step fails much more
severely than adding a later Layer segment, directly supporting a high-budget
early-denoise bucket.  Both expansions nevertheless cross the action gate, so
neither a monotonic timestep rule nor a monotonic layer-depth rule may replace
the measured Oracle/fallback policy.

The earlier individual-cell Q25 row is also rejected: its worst action cosine
is 0.998057 and worst relative L2 is 6.37%.  Its result, together with the
segment-aligned Q35 failure, establishes Q50 as the current full-segment
current-token floor.  These are three-request boundary results only; the
late4 candidate must next pass task-disjoint validation, video quality, GPU
exchange, and the final 100-request protocol.

Artifacts:

```text
dynamic_m1_m2/dynamic_budgets/20260831_guarded_segments_*/
dynamic_m1_m2/runtime/20260831_dynamic_m1_d163fff_pilot/
  comparison_guarded_segments_*_5eb04fb_droid_validation3*.json
```

## Task-disjoint validation18 gate

The promoted pilot frontier, `late4 x four segments x H50/Q50`, was replayed
on all 18 task-disjoint validation targets with three real history blocks per
target.  It passes only 14/18 action gates.  Failures are 5/6 early, 5/6
middle, and 4/6 late by stage; all three stages of
`validation_subset024_source018470` fail, as does
`validation_subset028_source020543_late`.

| Metric | Validation18 result | Gate | Status |
| --- | ---: | ---: | --- |
| action cosine mean/min | 0.999513 / 0.998022 | minimum >= 0.999 | fail |
| action rel-L2 mean/max | 3.04% / 7.51% | maximum <= 5% | fail |
| safe requests | 14/18 | 18/18 before promotion | fail |
| paired E2E geomean | 1.074x | final >= 1.35x mean target | fail |
| Sparse faster | 18/18 | >= 95% final | pass only |

The worst request is `validation_subset024_source018470_early`, with cosine
0.998022 and relative L2 7.51%.  This candidate therefore cannot enter the
100-request main result as a static table.  M1 must promote the demonstrated
risky requests/cells to a higher bucket or exact Dense fallback, and that gate
must be calibrated without using validation task identities.

The stable validation timing also corrects the three-request estimate.  Dense
validation18 averages 1.903 s and Sparse 1.773 s, whereas the pilot Dense file
contained a large request outlier.  The resulting paired CI is tight,
[1.069x, 1.079x], so the current 25% cell coverage is intrinsically below the
E2E target rather than merely noisy.

Artifact:

`dynamic_m1_m2/e2e/20260831_guarded_segments_late4_s4_h50q50_validation18/`

## Stable 108-request static frontier and dynamic-gate ceiling

The rejected late4/S4/H50Q50 table was completed across all task-disjoint
splits so its risk could supervise a dynamic promotion gate. Every target
still performs three real history calls and eight real DiTs per inference.

| Split | Requests | Paired geomean speedup | CI95 | Sparse faster | Action-safe |
| --- | ---: | ---: | --- | ---: | ---: |
| train | 72 | 1.0908x | [1.0874x, 1.0941x] | 72/72 | 61/72 |
| validation | 18 | 1.0738x | [1.0691x, 1.0786x] | 18/18 | 14/18 |
| test | 18 | 1.0867x | [1.0798x, 1.0938x] | 18/18 | 17/18 |

This is 108 paired target requests, but it is not the required main result:
the table fails 16 final-action requests and no GPU exchange has been applied
to this row. The training failures span early/middle/late as 6/1/4, so stage
alone cannot define the promotion.

The conservative M1 maximum-Head promotion implemented at `ecf7417` preserves
one shared Packed shape, but promotes every held-out candidate cell to Dense
because each cell has at least one fallback Head. A causal request-level
Gradient Boosting gate trained at `8962b0d` and selected with episode CV at
`e4529f5` catches all validation/test failures, but only retains the table on
3/18 requests per split. Its mixed speedups are 1.009x and 1.014x. Commit
`796a923` makes episode-CV risk, the 1.35x E2E target, and the 95%-faster target
mandatory; the resulting artifact is correctly rejected.

The data therefore rule out both extremes: any-fallback-Head promotion is
fully Dense, while a global request decision safe enough for held-out action
quality destroys coverage. The next budget representation must localize risk
more finely than a request-wide switch but use fewer shapes than the negative
3--4 group per-Head executor.
