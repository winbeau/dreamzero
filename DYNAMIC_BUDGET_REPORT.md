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
inside the packed stack is substantially more effective. The follow-up sweep
also identifies a required executor invariant: mutable current-token budgets
must stay fixed within a propagation segment. Segment-max budgets reach 0.9749
video cosine at 1.552x early-DiT speedup, while a 75% segment floor reaches
0.9970 at 1.378x. Confidence-controlled M1 runtime routing and a full eight-DiT
policy replay are still required before this phase can pass its quality gate.

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

All listed commits are pushed to
`origin/codex/dreamzero-anchor-sparse-opt`, and the H200 checkout is
fast-forwarded through `72a4c4d`.

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
