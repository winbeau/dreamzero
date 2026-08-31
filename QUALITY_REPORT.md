# Dynamic M1/M2 quality report

Date: 2026-08-31

## Status

This report records preliminary quality gates only. Full video evaluation,
worst-case task coverage, at least 100 paired requests, and closed-loop success
non-inferiority remain incomplete. No final quality claim is made.

## Full-budget invariants

The checkpoint validation path continues to require full-budget video, action,
and every-layer KV agreement with Dense. Sparse results may not reduce the
fixed eight real DiT evaluations inside the released 16-step scheduler.

## Balanced-policy eight-DiT smoke

One warmup and three paired measured real WebSocket requests were run with the
same generated observations, prompt, and seed. Dense used GPUs 2--3 and the
dynamic balanced Packed M2 policy used GPUs 5--6. Every request executed eight
real DiT evaluations.

| Metric | Result | Gate | Status |
| --- | ---: | ---: | --- |
| action cosine mean | 0.999634 | >= 0.999 | pass |
| action cosine minimum | 0.999485 | >= 0.999 | pass |
| action relative L2 mean | 2.696% | <= 5% | pass |
| action relative L2 maximum | 3.292% | <= 5% | pass |
| paired requests | 3 | >= 100 final | preliminary |

The worst measured action request was request index 1. Its action cosine was
0.999485 and relative L2 was 3.292%; it is retained in `comparison.json`
rather than hidden by the mean.

The service response exposes action but not the generated video tensor, so the
earlier single-timestep video gates are not replaced by this result. A full
trajectory video artifact/comparison path and closed-loop task evaluation are
still required.

Artifacts:

`dynamic_m1_m2/e2e/20260830_balanced_smoke/`

## Oracle-guarded shared-shape pilot

An Oracle-ranked shared timestep/layer table was evaluated on the same three
measured real-DROID history-chain requests used for the current runtime
diagnosis.  The first five real DiTs and every unselected layer remain Dense;
the selected 60/320 cells use 75% historical K/V and a swept current-token
budget.  All rows still execute eight real DiTs.

| Current Q/FFN in selected cells | Action cosine mean/min | Action rel-L2 mean/max | Preliminary gate |
| ---: | ---: | ---: | --- |
| 100% | 0.999775 / 0.999447 | 1.90% / 3.34% | pass |
| 75% | 0.999932 / 0.999907 | 1.17% / 1.38% | pass |
| 50% | 0.999754 / 0.999483 | 2.14% / 3.40% | pass |
| 35% | 0.999536 / 0.999400 | 3.06% / 3.63% | pass |

Q35 is the current pilot boundary and its worst request remains inside both
the 0.999 cosine and 5% relative-L2 thresholds.  This is materially safer than
the unguarded global or per-Head schedules, but the sample count is only three
and the table was selected from Oracle evidence.  It does not satisfy the
task-disjoint M1, video, 100-request, GPU-swap, or closed-loop gates, so no
final quality or non-inferiority claim is made.

Artifacts:

`dynamic_m1_m2/runtime/20260831_dynamic_m1_d163fff_pilot/`

## Real DROID history-chain gate

The candidate policies were next evaluated on real frames, robot state, and
instructions from the task-disjoint DROID Oracle subset. Each target was
preceded by three real history blocks so that errors could accumulate through
the same AR cache path used by deployment.

| Policy | Early cosine / rel-L2 | Middle cosine / rel-L2 | Late cosine / rel-L2 | Decision |
| --- | ---: | ---: | ---: | --- |
| balanced | 0.996445 / 8.43% | 0.990942 / 13.43% | 0.988943 / 16.10% | reject |
| quality | 0.997333 / 7.41% | 0.996679 / 8.25% | 0.981261 / 20.37% | reject |
| history floor 75% | 0.996397 / 8.84% | 0.993865 / 11.24% | 0.990773 / 13.95% | reject |
| Dense history | 0.996315 / 8.59% | 0.998383 / 5.77% | 0.997600 / 8.17% | reject |
| current floor 75% + Dense history | 0.998071 / 6.63% | 0.999405 / 4.05% | 0.997019 / 8.06% | reject late |
| full-budget Packed | 1.000000 / 0% | 1.000000 / 0% | 1.000000 / 0% | pass exactness |

The full-budget control proves that Packed M2 and service resets remain exact
under the real history chain. The approximation error grows by trajectory and
is not monotonic in a shared budget: raising late current compute from the
balanced table to the quality table made this late example worse. This is
direct evidence for confidence-aware M1 fallback and against selecting a
single global budget from isolated one-step averages.

No sparse row is promoted to the 100-request or closed-loop quality run yet.
The demonstrated late request is a mandatory Dense-fallback regression case.

Artifacts:

`dynamic_m1_m2/e2e/20260830_droid_108_round1/`

## Expanded task-disjoint quality gate

Across 72 train, 18 validation, and 18 test requests, the balanced profile
passes the action cosine/L2 gates for only 8/72, 0/18, and 2/18 requests. The
75%-current/Dense-history profile improves those counts to 36/72, 7/18, and
11/18, but still has worst relative L2 of 17.11%, 25.35%, and 28.59%.

Making the first two real DiT evaluations Dense does not solve accumulated
error: only 1/18 validation and 3/18 test requests pass, with test minimum
cosine 0.96760 and maximum relative L2 25.39%.

The corrected deployment-safe request gate avoids every final-action failure
only by falling back Dense on all 18 test requests. It therefore passes safety
but fails performance. The current global profile set must be replaced by a
finer dynamic executor policy before quality and acceleration can pass
together.

The validation-calibrated action-flow sentinel is also rejected. On untouched
test it misses one of 15 unsafe requests (false-sparse 6.67%) and triggers all
three safe requests. The missed early-stage request has action cosine 0.99358
and relative L2 11.48%, so this is a material safety failure rather than a
borderline threshold case.

## Two-group current-QKV checkpoint quality

| Candidate | Action cosine | Action rel-L2 | Video cosine | Video rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| trunk 50%, critical H100/Q50, normal H35/Q25 | 0.999908 | 1.451% | 0.9532 | 32.26% |
| trunk 35%, critical H100/Q35, normal H25/Q20 | 0.999901 | 1.543% | 0.8783 | 54.51% |

Full-budget video/action/cache exactness passes on both ranks. Critical-head
protection preserves the isolated action gate, but neither video result is
acceptable and isolated first-step quality cannot override the accumulated
DROID failures. No candidate is promoted.

## Dense action-history quality ablation

Keeping only the 25 action/state queries on complete historical K/V provides a
measurable but insufficient improvement. On the paired checkpoint input,
action cosine changes from 0.999832 to 0.999863 and relative L2 from 1.849% to
1.703%, while video cosine and relative L2 are unchanged within 0.002
percentage points.

Across the 18 task-disjoint validation targets, mean action cosine improves
from 0.995541 to 0.996276 and mean relative L2 falls from 9.293% to 8.377%.
Cosine improves on 11/18 requests and relative L2 improves on 12/18. However:

- zero of 18 requests satisfy cosine >=0.999 and relative L2 <=5%;
- minimum cosine is 0.977948;
- maximum relative L2 is 21.059%;
- the worst request remains
  `validation_subset024_source018470_late`, and it becomes worse than the
  original balanced result.

Dense action history is therefore not a monotonic safety mechanism and cannot
replace confidence promotion or Dense fallback. It is rejected as a global
policy despite improving the validation mean.

Artifact:

`dynamic_m1_m2/e2e/20260831_dense_action_history_balanced_validation18/`

## Dynamic action-history layer gate

The isolated checkpoint suggests that complete action history matters much
more in early Transformer layers: layers 1--13 improve action relative L2 to
1.515%, compared with 1.837% for layers 28--38, 1.703% for all packed layers,
and 1.849% without protection. Video error is unchanged and full-budget
video/action/cache exactness passes on both ranks.

The task-disjoint validation replay again rejects checkpoint-only selection.
Early-layer protection improves mean relative L2 only from 9.293% to 8.972%
and mean cosine from 0.995541 to 0.995675. It improves L2 on 9/18 requests and
cosine on 8/18, but zero requests satisfy both final-action gates. Minimum
cosine is 0.977018 and maximum relative L2 is 21.385% on
`validation_subset024_source018470_late`.

The result supports layer-dependent routing but not the specific global
early-layer schedule. M1 still needs request/head confidence and exact Dense
fallback for the demonstrated regression case.

The same layer bucket at DiT index 4 improves the isolated action relative L2
from 2.487% to 1.457%, so late denoising also benefits. This rules out a simple
"protect action history only on early DiT steps" explanation. The validation
failure despite per-step action improvement points to accumulated video/cache
state as the next quality target.

Raising early-layer video-query history to 75% also fails to give monotonic
trajectory recovery. The isolated checkpoint improves to action/video relative
L2 of 1.259%/8.525%, but validation18 mean action L2 worsens to 10.011% and the
worst request reaches 23.406%. Two requests pass, one more than the union
already covered by the conservative profile, but ten of 18 still require
Dense execution even under a perfect selector over all measured profiles.

## Dense-suffix recovery quality gate

The checkpoint gate suggests a clean video benefit from Dense recovery:
suffix one, three, and five reduce video relative L2 from 8.758% to 7.512% to
6.708%.  Final action quality does not follow this ordering.  Validation18
suffix three reaches mean cosine 0.995709 and mean relative L2 8.999%, with
only request index 4 satisfying both action gates.  Its minimum cosine is
0.978420 and maximum relative L2 is 20.828%.

Suffix five is worse: mean cosine 0.995497, mean relative L2 9.103%, minimum
cosine 0.976171, maximum relative L2 21.758%, and zero safe requests.  The
worst request remains `validation_subset024_source018470_late` for both rows.
Thus final Dense layers can cosmetically repair video output without undoing
the hidden-state/action error accumulated through earlier packed segments.
Suffix recovery is rejected as a fallback; low-confidence routes still need
earlier budget promotion or exact Dense execution.

## Propagation-radius quality gate

Increasing propagation frequency from radius two/every five to radius
two/every three does not improve the released checkpoint: action/video
relative L2 changes from 1.849%/8.758% to 1.863%/8.830%.  Increasing spatial
radius to three is locally much better for video, reaching cosine 0.999155 and
4.233% relative L2, while action remains at 0.999834 cosine and 1.839% L2.

The validation18 trajectory disproves video-output recovery as an action
safety proxy.  Radius three/every five has mean action cosine 0.995300, minimum
cosine 0.979434, mean relative L2 9.553%, maximum L2 20.289%, and zero safe
requests.  Its worst request is again
`validation_subset024_source018470_late`.  Spatial interpolation can make the
video tensor substantially closer to Dense while the action registers retain
earlier packed-state error.  Radius three is therefore rejected as a global
quality policy despite its strong checkpoint video row.

## Maximum-current action-readout quality gate

The maximum packed current prefix was tested as an extra K/V source for only
the 25 action/state queries. Identical schedules produce exactly the same
quality numbers after exchanging GPUs, so the comparison is deterministic:

| Schedule | Action cosine | Action rel-L2 | Change from none |
| --- | ---: | ---: | ---: |
| none | 0.99983209 | 1.84871% | -- |
| segment entries | 0.99983412 | 1.84485% | -0.00387 pp |
| segment exits | 0.99982333 | 1.93281% | +0.08410 pp |
| all packed layers | 0.99978894 | 2.14739% | +0.29868 pp |

Video relative L2 remains between 8.753% and 8.758% for all rows, confirming
that the intervention is restricted to the action readout. The sign changes
with hidden-state freshness: segment-entry tokens were just recovered and
repacked, while inactive tail tokens at an exit have skipped up to four
Transformer blocks. The latter are not valid current features even though
their storage remains allocated.

The best row improves action L2 by only 0.21% relative, far below the roughly
9% mean and 18--20% tail errors observed in complete validation trajectories.
It is therefore rejected before validation18. Dynamic M1 may not use stale
maximum-prefix visibility as confidence fallback; any future within-segment
recovery must update the exposed token state or select the exact Dense path.

Artifact:

`dynamic_m1_m2/dynamic_budgets/20260831_max_action_current/`

## Propagation-aligned guarded boundary

Complete propagation segments were replayed on the three-request real-DROID
pilot so that the configured current budget exactly matches executed Packed
compute.  The accepted and rejected boundaries are:

| Schedule | Action cosine mean/min | Action rel-L2 mean/max | Decision |
| --- | ---: | ---: | --- |
| late3, 4 segments, H75/Q50 | 0.999695 / 0.999317 | 2.28% / 3.87% | preliminary pass |
| late3, 4 segments, H75/Q35 | 0.998594 / 0.996235 | 4.27% / 8.72% | reject |
| late3, 4 segments, H50/Q50 | 0.999700 / 0.999375 | 2.44% / 3.93% | preliminary pass |
| late4, 4 segments, H50/Q50 | 0.999569 / 0.999218 | 2.92% / 4.09% | preliminary pass |
| late5, 4 segments, H50/Q50 | 0.994347 / 0.984626 | 8.86% / 18.41% | reject |
| late4, 5 segments, H50/Q50 | 0.999158 / 0.998245 | 3.78% / 5.92% | reject |

The result supports the early-denoise hypothesis: adding DiT index 3 causes
the largest failure even though its Dense-Oracle aggregate score is adjacent
to the accepted late bucket.  It also falsifies a simple "deeper is always
safer" layer rule because adding layers 21--25 crosses both worst-case action
gates.  Q35 is rejected once its full requested coverage is actually executed;
the earlier scattered-cell Q35 pass was not evidence for a full-segment Q35
policy.

`late4 x four segments x H50/Q50` is promoted only to task-disjoint validation,
not to a final quality claim.  The sample count is three, generated video is
not yet compared on this service path, and confidence fallback, 100-request,
GPU-exchange, and closed-loop gates remain open.

## Propagation-aligned validation18 result

The task-disjoint validation replay rejects the pilot frontier.  Four of 18
requests fail the joint cosine/L2 gate:

| Stage | Request | Cosine | Relative L2 |
| --- | --- | ---: | ---: |
| early | `validation_subset024_source018470_early` | 0.998022 | 7.51% |
| middle | `validation_subset024_source018470_middle` | 0.998592 | 5.79% |
| late | `validation_subset024_source018470_late` | 0.998624 | 5.39% |
| late | `validation_subset028_source020543_late` | 0.999206 | 5.24% |

Overall action cosine is 0.999513 mean and 0.998022 minimum; relative L2 is
3.04% mean and 7.51% maximum.  Stage pass counts are 5/6 early, 5/6 middle,
and 4/6 late.  The failure across all three stages of one held-out trajectory
shows that a timestep/layer table alone cannot express request-conditioned
risk.  Static late4/H50/Q50 is rejected before test, 100-request, video, or
closed-loop promotion.

The four failures become mandatory M1 false-sparse regression cases.  A future
confidence gate must promote them using task-disjoint runtime features, not
their task or trajectory identity, while preserving the 14 already-safe
requests whenever confidence is calibrated.

Artifact:

`dynamic_m1_m2/e2e/20260831_guarded_segments_late4_s4_h50q50_validation18/`
