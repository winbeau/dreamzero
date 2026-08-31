# Dynamic M1 classifier report

Date: 2026-08-30

## Status

The earlier v2 statistical result is now classified as a contaminated
ablation, not a deployment-safe M1 result. A feature audit found that it used
ground-truth DROID action magnitude/variation and offline trajectory-stage,
trajectory-length, and instruction-position annotations. Those values are
available in the Oracle dataset but are not available to the online router at
decision time. The old bundle must therefore not support a paper Claim.

The training and request-gate feature contracts now exclude those fields. The
corrected v3 per-head classifier has been retrained and passes its statistical
gates. The complete M1 phase remains open because the corrected request-level
gate and real dynamic-routing final-action replay are not yet accepted. The
prior v2 metrics below are retained only to document the superseded ablation.

Implementation commits:

- `6fda0f6`: leakage-safe task-disjoint classifier pipeline;
- `14b3f40`, `8bba7e0`: real Oracle parquet integration and schema fixes;
- `3e8e4fb`: persisted deployment prior table and bundle metadata;
- `2b8d64d`: calibrated fallback and minimum-macro-F1 model selection.
- `c577622`: portable runtime `RoutePolicy` type for direct bundle loading.
- `4ce658e`: request-level final-action safety gate from first-two-DiT state.
- `df04445`: distinguish request-gate safety from performance acceptance.
- `1510f4b`: remove future-action and offline trajectory annotations from M1.

All listed commits are pushed to `origin/codex/dreamzero-anchor-sparse-opt` and
the H200 checkout was fast-forwarded through them.

The superseded v2 bundle encoded `RoutePolicy` under the training script's
`__main__` module and therefore could not be loaded by a clean deployment
process. It has been losslessly migrated to schema v2 and verified in a fresh
Python process at:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/m1_classifier/20260830_full_v2_calibrated/
    selected_m1_bundle_v2_portable.joblib
```

The migrated artifact retains the former estimator, confidence calibrator,
threshold `0.9598636879969709`, zero mechanical bucket promotion, and all
12,800 `(t,l,h)` train-only prior rows.

## Data and split protocol

The input contains 1,382,400 conservative `(request,timestep,layer,head)` rows
from the complete q32 Oracle. Source episodes, rather than individual rows,
form the immutable split:

| Split | Episodes | Rows |
| --- | ---: | ---: |
| train | 24 | 921,600 |
| validation | 6 | 230,400 |
| test | 6 | 230,400 |

The label is the minimum safe bucket from `[10, 20, 25, 35, 50, 75, 100]%`,
conservatively maximized across video/action queries and conditional/
unconditional CFG branches. Under-predicting the Oracle bucket receives 20x
cost in the cost-sensitive candidate.

## Deployment-feature audit

The corrected M1 feature set uses timestep position, scheduler value,
layer/head position, current robot-state magnitudes, historical support
turnover, historical attention concentration/entropy, historical Qa-to-Kv
versus Qv-to-Kv correlation, the previous two VV changes, VV acceleration,
and leave-one-episode-out train priors for `(t,l,h)`.

The following Oracle-only request fields are now forbidden as model inputs:

- ground-truth action L2, standard deviation, and temporal-delta L2;
- offline trajectory stage and fraction;
- complete trajectory length and length bucket;
- offline instruction-position annotation.

Current-call Dense attention metrics are explicitly forbidden. In particular,
current support turnover, current VV change, current entropy, current maximum
mass, current Qa/Qv correlation, and all current worst-query quality metrics
cannot enter the feature matrix. Missing temporal history remains missing at
the first two DiT evaluations and is handled by the fitted imputer; it is not
filled with current Oracle evidence.

## Corrected deployment-safe v3 result

The complete candidate comparison was rerun on the same task-disjoint rows,
with 200 task-level bootstrap repeats and no future-action or offline
trajectory metadata in the feature matrix.

| Candidate | Test macro-F1 | Test false-sparse | Mean keep | Confidence fallback | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| original 3-component GMM | 0.061 | 0.000% | 100.00% | 0.00% | reject: degenerate Dense |
| supervised logistic | 0.191 | 0.260% | 83.90% | 0.00% | reject: no confidence fallback |
| Gradient Boosting | 0.570 | 1.026% | 83.52% | 33.29% | reject: false-sparse >1% |
| small MLP | 0.160 | 0.197% | 84.90% | 0.00% | reject: macro-F1 and no fallback |
| cost-sensitive Gradient Boosting | 0.557 | 0.694% | 83.92% | 33.25% | selected |

Selected-v3 test gates:

| Metric | Result | Gate | Status |
| --- | ---: | ---: | --- |
| false-sparse rate | 0.694% | <1% | pass |
| macro-F1 | 0.557 | >=0.50 | pass |
| p05 mass >=0.90 rate | 99.984% | >=95% | pass |
| local attention-output gate | 99.310% | diagnostic | recorded |
| confidence fallback rate | 33.250% | >0% | pass |
| Dense route rate | 56.039% | diagnostic | recorded |
| route-confidence ECE | 0.03562 | calibrated | pass |

The 200-repeat bootstrap gives false-sparse 0.689% with 95% interval
[0.570%, 0.805%], mean keep 83.918% [83.690%, 84.102%], and mass-retention
rate 99.983% [99.956%, 99.998%]. The test route counts are 123,362 critical,
76,608 uncertain, 21,688 slow-changing, 8,706 stable, and 36
predictable-late head states.

The corrected artifact is:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/m1_classifier/20260831_deployment_safe_v3/
```

Its `summary.json` deliberately reports `passed: false`: classifier statistics
do not substitute for the required final-action and closed-loop gates.

## Corrected request-level safety gate

The request-level gate was regenerated from the corrected v3 per-head bundle.
It sees only robot-state magnitudes and M1/history features available after the
first two mandatory real DiT evaluations; offline stage, fraction, and complete
trajectory length are absent.

The selected cost-sensitive logistic rule is safe but collapses to Dense on
the untouched test split:

| Metric | Validation | Test | Gate |
| --- | ---: | ---: | --- |
| false-sparse rate | 0% | 0% | safety pass |
| Dense fallback rate | 94.44% | 100.00% | diagnostic |
| final-action quality failures | 0 / 18 | 0 / 18 | safety pass |
| mixed end-to-end speedup | 1.009x | 1.000x | performance fail |
| strictly faster requests | 5.56% | 0.00% | performance fail |

The result correctly reports `safety_gates_passed: true`,
`performance_gates_passed: false`, and `passed: false`.

More importantly, an Oracle that chooses the fastest quality-safe profile for
every request reaches only 1.1085x/1.0683x/1.1254x mixed end-to-end speedup on
train/validation/test. Thus no classifier over the current three global
profiles (`balanced`, `conservative`, `dense`) can reach the 1.35x target.
This is a profile-family ceiling, not merely a classifier error.

Artifact:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/request_gate/20260831_deployment_safe_v3/
```

## Candidate comparison

The following numbers describe the superseded feature-contaminated v2 run.
They are preserved for reproducibility, not accepted as deployment evidence.

The original GMM and supervised logistic control were measured in the first
complete run. The calibrated v2 run re-evaluated the three statistically
credible nonlinear candidates after making actual confidence fallback and
macro-F1 mandatory for selection.

| Candidate | Test macro-F1 | Test false-sparse | Mean keep | Confidence fallback | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| original 3-component GMM | 0.061 | 0.000% | 100.00% | degenerate Dense | reject |
| supervised logistic | 0.190 | 0.253% | 83.93% | 0.00% | reject: no calibrated fallback |
| Gradient Boosting | 0.559 | 0.931% | 84.00% | 34.45% | feasible, not selected |
| small MLP | 0.516 | 1.201% | 84.48% | 33.62% | reject: false-sparse >1% |
| cost-sensitive Gradient Boosting | 0.549 | 0.637% | 84.33% | 34.53% | selected |

The selected model is not the sparsest point estimate. It is chosen because
validation false-sparse risk is minimized first, then mean keep ratio, then
macro-F1, subject to all hard gates. This prevents the unsafe logistic model
from winning only because every prediction is mechanically promoted without a
real uncertainty fallback.

## Superseded v2 test gates

| Metric | Result | Gate | Status |
| --- | ---: | ---: | --- |
| false-sparse rate | 0.637% | <1% | pass |
| critical false-sparse rate | 0.721% | diagnostic | recorded |
| macro-F1 | 0.549 | >=0.50 | pass |
| p05 mass >=0.90 rate | 99.980% | >=95% | pass |
| local attention-output gate | 99.366% | diagnostic | recorded |
| confidence fallback rate | 34.528% | >0% | pass |
| Dense route rate | 57.153% | diagnostic | recorded |
| route-confidence ECE | 0.03696 | calibrated | pass |

The 200-repeat source-episode bootstrap gives:

| Metric | Mean | 95% CI |
| --- | ---: | ---: |
| false-sparse rate | 0.633% | [0.514%, 0.753%] |
| mean keep ratio | 84.334% | [84.160%, 84.523%] |
| p05 mass >=0.90 rate | 99.980% | [99.947%, 99.997%] |

## Superseded route semantics

The deployment output is more than one budget label. It maps each head state
to `critical`, `stable`, `slow-changing`, `predictable-late`, or `uncertain`,
plus calibrated confidence, a nested anchor-profile prefix, refresh frequency,
linear-extrapolation permission, and Dense fallback.

On the test split, 79,553 of 230,400 head states are uncertain and fall back
to Dense. Only 36 states satisfy the deliberately strict predictable-late
rule; this category will not be enlarged until VV extrapolation and sentinel
experiments provide direct evidence.

## Superseded artifact

The model bundle, prior table, per-candidate metrics, and summary are outside
Git at:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/m1_classifier/20260830_full_v2_calibrated/
```

The selected bundle is about 2.4 MiB and the 12,800-row deployment prior table
is about 69 KiB. No Oracle table, checkpoint, trace, or fitted binary is added
to Git.

## Remaining M1 work

The same-noise downstream pilot in `DYNAMIC_ORACLE_REPORT.md` changes the M1
supervision requirement. A head's local mass/output budget cannot be used as a
standalone criticality label: locally highest-budget head 14 and lowest-budget
head 28 have similar final-action sensitivity at DiT 0/layer 39, and layer 20
is more sensitive than both layer 0 and layer 39 for head 14. Query-path
importance also changes across trajectory stage. The next classifier revision
must therefore include calibrated downstream action sensitivity or a
conservative proxy trained against it, while retaining Dense fallback for
unscanned/uncertain cells.

- integrate the selected bundle into timestep/layer/head-group budget routing;
- measure actual route/classifier overhead on GPU;
- replace the current global profile family with finer shared-group dynamic
  budgets; request-level selection alone has an Oracle ceiling below target;
- replay held-out requests through the real DreamZero policy;
- require final action cosine >=0.999 and relative L2 <=5%;
- expand same-noise downstream labels across task-disjoint shared head groups,
  save the worst false-sparse/fallback cases, and connect them to final
  action/video changes;
- recheck calibration after fixed-shape head-group quantization.
