# Dynamic M1 classifier report

Date: 2026-08-31

## Status

The shared-Packed deployment branch is now closed as a negative result. The
local per-Head classifier is statistically useful, but maximum-Head promotion
makes every held-out candidate cell Dense. A causal request-wide gate retains
only 3/18 sparse routes per held-out split, realizes about `1.01x`, and has
6/72 false-sparse train episode-CV events. It is explicitly rejected and not
deployed. `FINAL_STATUS.md` freezes the decision and next route.

The earlier v2 statistical result is now classified as a contaminated
ablation, not a deployment-safe M1 result. A feature audit found that it used
ground-truth DROID action magnitude/variation and offline trajectory-stage,
trajectory-length, and instruction-position annotations. Those values are
available in the Oracle dataset but are not available to the online router at
decision time. The old bundle must therefore not support a paper Claim.

The training and request-gate feature contracts now exclude those fields. The
corrected v3 per-head classifier has been retrained and passes its statistical
gates. A risk-controlled four-shape grouped router is now implemented, but the
complete M1 phase remains open because its task-disjoint downstream risk table
has not yet been populated, the grouped v3 artifact has not yet been
re-evaluated, and real dynamic-routing final-action replay is not yet accepted.
The prior v2 metrics below are retained only to document the superseded
ablation.

Implementation commits:

- `6fda0f6`: leakage-safe task-disjoint classifier pipeline;
- `14b3f40`, `8bba7e0`: real Oracle parquet integration and schema fixes;
- `3e8e4fb`: persisted deployment prior table and bundle metadata;
- `2b8d64d`: calibrated fallback and minimum-macro-F1 model selection.
- `c577622`: portable runtime `RoutePolicy` type for direct bundle loading.
- `4ce658e`: request-level final-action safety gate from first-two-DiT state.
- `df04445`: distinguish request-gate safety from performance acceptance.
- `1510f4b`: remove future-action and offline trajectory annotations from M1.
- `22a8a3d`: risk-controlled four-bucket Head grouping, downstream coverage
  table, and post-quantization task-disjoint evaluator.
- `fa1d945`: request-local causal feature state, explicit online observation
  schema contract, first-two-DiT Dense policy, and feature-provenance fallback.

All listed commits are pushed to `origin/codex/dreamzero-anchor-sparse-opt`.
The H200 checkout was previously fast-forwarded through `1510f4b`; `22a8a3d`
has not yet been synchronized because both configured SSH routes were
unavailable.

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

## Risk-controlled fixed-shape Head grouping

Commit `22a8a3d` replaces the static train-prior bridge as the intended M1
deployment contract. It loads the calibrated v3 estimator, confidence
calibrator, promotion policy, and feature order, then conservatively rounds
the seven M1 ratios upward into the four executor shapes `[25, 50, 75, 100]%`.
Heads sharing one effective ratio form one execution group, so each
request/timestep/layer uses at most four FA2 shapes. Historical K/V and current
Q/K use the same group ratio in this first deployable contract; action/state
tokens remain Dense in M2.

The grouped decision retains all requested M1 semantics per Head:

- calibrated route confidence and the original classifier fallback;
- `critical`, `stable`, `slow-changing`, `predictable-late`, or `uncertain`;
- the nested anchor-prefix ratio and route refresh frequency;
- late-step extrapolation permission only for confident, two-history,
  low-turnover, low-VV-change routes.

Feature provenance and downstream safety are separate mandatory gates, not
additional confidence features. `build_downstream_head_risk_table.py` accepts only scale-zero
shared-group removals with exact trace agreement and one application. A Head
is marked scanned only after the configured task-disjoint split, all required
trajectory stages, and a minimum unique-request count are covered. Action and
video thresholds are explicit inputs. Failed evidence marks every Head in the
removed group unsafe; missing coverage stays unknown. Classifier-low-
confidence, feature-contract, downstream-unsafe, and downstream-unknown
fallbacks are logged separately and all force Dense.

`evaluate_dynamic_m1_group_router.py` replays the frozen bundle on validation
and test without retuning, rounds Oracle labels upward to the same four
executor buckets, and reports post-grouping macro-F1, false-sparse rate,
confusion matrix, mass retention, calibration, group counts, fallback causes,
and 200-repeat source-episode bootstrap. Its output deliberately remains
`passed: false` until action/video policy replay and closed-loop gates pass.

Local implementation gates are now 27 passing online-state/grouped-M1/
classifier/dynamic-budget tests, Ruff, Python compilation, and
`git diff --check`. No post-grouping v3 number is
reported yet: the required video-enabled downstream risk artifact lives on
the currently unreachable H200 server, and treating every unscanned cell as
safe would violate the M1 risk contract.

## Online causal feature contract

Commit `fa1d945` closes a correctness gap between the offline v3 table and the
runtime router. The v3 historical features were computed from sampled Dense
attention probabilities and VV outputs; the current Packed path did not yet
produce observations with those exact semantics. Loading the estimator alone
would therefore have made unavailable Oracle signals appear deployable.

`OnlineM1FeatureState` now owns request-local causal history. It enforces
routing before completion of each of the eight real DiT evaluations, forces
the first two real DiTs Dense, and accepts only observations whose declared
schema matches the bundle's `online_observation_schema`. A missing observer,
missing adjacent history, or schema mismatch is represented by a dedicated
per-Head feature fallback and forces the grouped executor to 100%. Missing
observations may still advance the real-DiT state, so instrumentation failure
degrades to Dense instead of corrupting the denoising-step sequence.

The existing deployment-safe-v3 artifact intentionally has no online
observation schema. It therefore routes fully Dense under this contract until
a lightweight Packed observer is implemented, its proxy features are
collected on real requests, and a matching bundle is retrained. This is a
safety result, not a speed result.

## Packed-path causal proxy implementation

Commits `287a54e` and `287f8a8` implement the missing deployable observation
path under the deliberately incompatible schema
`dreamzero-packed-m1-proxy-v2`. It does
not reconstruct Dense video attention or reuse the offline Oracle feature
names. Each real DiT records only signals already available to the executor:

- the per-Head mean/RMS signature of the 25 Dense action/state register
  attention outputs before O projection;
- adjacent-real-DiT register-output relative L2 and cosine;
- conditional/unconditional disagreement and signature norm;
- action-conditioned Router support turnover, normalized entropy, and maximum
  mass when the lightweight Router is actually refreshed.

The grouped Packed path reconstructs only the small `[B, 25, 40, head_dim]`
register output from fixed-shape Head groups for observation. It does not
scatter the video sequence. The observer advances only when the action head
executes one of the eight real DiT calls; skipped scheduler steps do not
advance history, and a Dense sentinel rerun overwrites the earlier sparse
signature for the same DiT. Request start clears stale DiT indices and the
last step is finalized at the existing request flush boundary.

The DROID collector fixes 16 scheduler steps and eight real DiT calls, attaches
the observer only after history warmup, and saves a reduced float32
`[dit, layer, head, metric]` NPZ rather than activations. For teacher-feature
collection it refreshes only the lightweight Router on every real DiT. Cached
routes are never reported as artificial zero turnover. The NumPy-only artifact
loader and strict one-to-one merge causally shift observation `t-1` into the
Oracle-labelled row at `t`; a matching classifier mode excludes every legacy
Dense `previous_vv_*` input and embeds the proxy schema into the fitted bundle.

Local gates after the implementation commit are:

| Gate | Result |
| --- | ---: |
| Packed observer/model integration | 32/32 passed |
| online M1/group router/classifier/budget/merge | 30/30 passed |
| proxy-schema training smoke test | passed; no Dense-history feature leakage |
| Ruff, Python compilation, `git diff --check` | passed |

H200 connectivity and the reserved Dense 2--3 / Sparse 5--6 services were
recovered after these local gates. Real proxy collection, overhead timing,
task-disjoint retraining, and held-out quality replay are not yet claimed in
this section; they begin only after the implementation/report commits are
pushed and the H200 checkout is fast-forwarded through the existing remote.

### H200 proxy smoke and route-confidence calibration

The first two-request H200 smoke on commit `287a54e` passed the structural
gate on both GPUs 2 and 3: every request executed 16 scheduler steps and eight
real DiT calls, produced eight valid `40 x 40` observations, retained the
expected first-step missing history, and had finite later history. Target
request latency was 3.63--3.72 seconds in the independent one-GPU collector,
peak allocated memory was 61.62 GiB, and each compressed observation artifact
was about 188 KiB.

The smoke also exposed a semantic failure before the first full collection
was accepted. Directly softmaxing raw cosine Router scores yielded normalized
entropy near `0.99996` and maximum mass near `0.0012` on both requests, so the
two proposed concentration features were almost constant despite non-trivial
support turnover (`0.40--1.00`). The partial v1 collection was stopped and
retained as superseded evidence rather than merged into training data.

Commit `287f8a8` standardizes Router scores independently within each frame
before computing entropy and maximum mass, while leaving support selection and
turnover on the original scores. Because this changes feature semantics, the
artifact/bundle schema is bumped to `dreamzero-packed-m1-proxy-v2`; the loader
rejects all v1 artifacts. The revised route metric has explicit uniform and
peaked-distribution tests, and the complete local gate remains 32/32 Torch plus
30/30 CPU tests with Ruff and `git diff --check` passing.

Superseded pilot artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/m1_proxy/20260831_proxy_v1_smoke_287a54e/
  dynamic_m1_m2/m1_proxy/20260831_proxy_v1_full_287a54e/
```

## Full Packed-proxy-v2 collection and classifier

The accepted H200 collection covers all 108 task-disjoint requests (72 train,
18 validation, 18 test) on GPUs 2 and 3, with 54 requests per worker. Global
audit found 108 unique artifacts, no duplicates, no missing or extra Oracle
request keys, and no structural gate failure. Every artifact has eight real
DiT observations with `40 x 40` layer/Head geometry. Mean target-request time
in this independent one-GPU collection was 3.6816 seconds (p50 3.6923, p95
3.7385), maximum allocated memory was 61.6182 GiB, and the strict merge added
the causal proxy features to all 1,382,400 Oracle-labelled rows one-to-one.

The aggregate proxy statistics do not support a hard-coded monotonic late-step
rule. Median per-request action-register output relative L2 across real DiTs
1--7 is `[0.336, 0.337, 0.456, 0.397, 0.441, 0.380, 0.478]`; it does not
decrease late. Conditional/unconditional disagreement does decrease from
0.110 at DiT 0 to 0.071 at DiT 7, while support turnover remains high and
non-monotonic. This is direct evidence for supervised per-state routing and
fallback rather than a fixed "late is sparse" table.

Commit `07e0a58` versions the classifier mode as `packed-proxy-v2`, preventing
the superseded v1 name from appearing in a v2 bundle. All five requested model
families were trained with task-disjoint splits and the frozen validation
policy:

| Candidate | Test macro-F1 | Test false-sparse | Mean keep | Confidence fallback | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| original 3-component GMM | 0.0613 | 0.000% | 100.00% | 0.00% | reject: degenerate Dense |
| supervised logistic | 0.1919 | 0.291% | 83.93% | 0.00% | reject: no confidence fallback |
| Gradient Boosting | 0.5643 | 1.067% | 83.70% | 33.82% | reject: false-sparse >1% |
| small MLP | 0.1612 | 0.208% | 84.89% | 0.00% | reject: no confidence fallback |
| cost-sensitive Gradient Boosting | 0.5589 | 0.720% | 83.87% | 33.38% | selected |

The selected v2 classifier retains 99.9787% of held-out rows at or above 0.9
Dense mass and passes the local attention-output gate on 99.2839% of rows. Its
200-repeat task-level bootstrap is:

| Metric | Mean | 95% CI |
| --- | ---: | ---: |
| false-sparse rate | 0.7141% | [0.5793%, 0.8338%] |
| mean keep ratio | 83.8672% | [83.6535%, 84.0314%] |
| p05 mass >=0.90 rate | 99.9787% | [99.9470%, 99.9957%] |

The statistical gates pass, but the classifier summary correctly remains
`passed: false` pending actual DreamZero action/video replay. The strict route
semantics currently produce 122,909 critical, 76,898 uncertain, and 30,593
slow-changing states; no state satisfies the stable or predictable-late rule.
Therefore v2 grants no linear-extrapolation permission yet.

Accepted artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/m1_proxy/20260831_proxy_v2_full_287f8a8/
  dynamic_m1_m2/m1_classifier/20260831_packed_proxy_v2_07e0a58/
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

- collect the implemented Packed proxy schema on real requests, retrain M1
  against that schema, and connect the resulting causal decisions to the
  model's per-DiT Packed-M2 table update;
- measure actual route/classifier overhead on GPU;
- replace the current global profile family with finer shared-group dynamic
  budgets; request-level selection alone has an Oracle ceiling below target;
- replay held-out requests through the real DreamZero policy;
- require final action cosine >=0.999 and relative L2 <=5%;
- expand same-noise downstream labels across task-disjoint shared head groups,
  save the worst false-sparse/fallback cases, and connect them to final
  action/video changes;
- populate the task-disjoint downstream risk table and run the implemented
  fixed-shape post-quantization calibration evaluator.

## Shared Packed promotion gate and rejection

Commits `ecf7417` and `eccc5fe` add a conservative bridge from online M1 to a
shared timestep/layer Packed table. The Oracle table supplies the base H/Q
bucket, M1 may only promote it, and the executor launches no per-Head group in
this mode. Current-token budgets remain constant within each five-layer
propagation segment. Full-budget cells stay on the exact Dense path, while a
fallback Head promotes its layer through the same fixed bucket interface.

This maximum-Head rule is safe but degenerate for the current Packed-proxy-v2
classifier. Offline replay on both validation18 and test18 shows that every
candidate `(request, DiT>=4, layer 1--20)` cell contains at least one Dense
fallback Head. Each such cell has 6--32 fallback Heads, averaging 16.33 on
validation and 16.12 on test. Consequently 100% of candidate cells promote
to Dense and zero of 36 held-out requests retain any sparse cell. The live
6105 service was not restarted for a result that was analytically certain to
be fully Dense.

Commit `547f8b8` also fixes the post-grouping evaluator to honor the feature
schema stored in the selected bundle. Before this fix,
`evaluate_dynamic_m1_group_router.py` always requested Dense-Oracle columns
and passed the default Dense feature list to `sequential_predict`, so a
Packed-proxy bundle could not be evaluated by the advertised CLI.

The static late4/S4/H50Q50 table was then replayed on train72 and test18 to
obtain final-action supervision for a request-level shared-budget promotion
gate. The base-table unsafe labels are 11/72 train, 4/18 validation, and 1/18
test. Commits `8962b0d`, `e4529f5`, and `796a923` add a causal binary
`shared_sparse` versus `dense_fallback` trainer. Its features come only from
robot state and the Packed observer/M1 state available before the third real
DiT; task identity, trajectory stage/length, future DiTs, current Dense
attention, and final action are excluded. Model selection uses train
source-episode leave-one-out predictions plus validation thresholds; test is
report-only.

| Candidate | Train episode-CV false-sparse | Validation false-sparse / Sparse route | Test false-sparse / Sparse route | Decision |
| --- | ---: | ---: | ---: | --- |
| cost-sensitive logistic | 7/72 | 0/18 / 3/18 | 1/18 / 1/18 | reject safety |
| cost-sensitive Gradient Boosting | 6/72 | 0/18 / 3/18 | 0/18 / 3/18 | selected diagnostic; reject CV |
| cost-sensitive small MLP | 8/72 | 0/18 / 1/18 | 0/18 / 0/18 | reject degenerate Dense |

The selected Gradient Boosting gate has zero final-action failures on the
validation and test mixed routes, but it fails the episode-level safety gate
and retains too little sparse coverage. Mixed speedup is only 1.009x on
validation and 1.014x on test, with 16.7% of requests strictly faster. The
artifact therefore records `safety_gates_passed: false`,
`performance_gates_passed: false`, and `passed: false`. This candidate is not
connected to the live server and cannot support an M1 Claim.

Artifacts:

```text
dynamic_m1_m2/request_gate/20260831_shared_h50q50_proxy_v2_episode_cv/
dynamic_m1_m2/e2e/20260831_guarded_segments_late4_s4_h50q50_train72/
dynamic_m1_m2/e2e/20260831_guarded_segments_late4_s4_h50q50_test18/
```

Stage report commit: `de92d03` (pushed to
`origin/codex/dreamzero-anchor-sparse-opt`).
