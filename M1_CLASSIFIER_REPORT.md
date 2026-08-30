# Dynamic M1 classifier report

Date: 2026-08-30

## Status

The leakage-safe statistical M1 gate is complete. A calibrated,
cost-sensitive HistGradientBoosting classifier is selected on the validation
split and passes the untouched test-set false-sparse, mass-retention,
macro-F1, confidence-fallback, and 200-repeat task-bootstrap requirements.

The complete M1 phase remains open until the selected policy is replayed
through DreamZero and achieves final action cosine >=0.999. Local Oracle
attention-output metrics are not substituted for that model-level gate.

Implementation commits:

- `6fda0f6`: leakage-safe task-disjoint classifier pipeline;
- `14b3f40`, `8bba7e0`: real Oracle parquet integration and schema fixes;
- `3e8e4fb`: persisted deployment prior table and bundle metadata;
- `2b8d64d`: calibrated fallback and minimum-macro-F1 model selection.

All listed commits are pushed to `origin/codex/dreamzero-anchor-sparse-opt` and
the H200 checkout was fast-forwarded through them.

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

## Deployment-safe features

M1 uses timestep position, scheduler value, layer/head position, trajectory and
instruction metadata, action/state magnitudes, historical support turnover,
historical attention concentration/entropy, historical Qa-to-Kv versus
Qv-to-Kv correlation, the previous two VV changes, VV acceleration, and
leave-one-episode-out train priors for `(t,l,h)`.

Current-call Dense attention metrics are explicitly forbidden. In particular,
current support turnover, current VV change, current entropy, current maximum
mass, current Qa/Qv correlation, and all current worst-query quality metrics
cannot enter the feature matrix. Missing temporal history remains missing at
the first two DiT evaluations and is handled by the fitted imputer; it is not
filled with current Oracle evidence.

## Candidate comparison

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

## Selected-policy test gates

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

## Route semantics

The deployment output is more than one budget label. It maps each head state
to `critical`, `stable`, `slow-changing`, `predictable-late`, or `uncertain`,
plus calibrated confidence, a nested anchor-profile prefix, refresh frequency,
linear-extrapolation permission, and Dense fallback.

On the test split, 79,553 of 230,400 head states are uncertain and fall back
to Dense. Only 36 states satisfy the deliberately strict predictable-late
rule; this category will not be enlarged until VV extrapolation and sentinel
experiments provide direct evidence.

## Artifact

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

- integrate the selected bundle into timestep/layer/head-group budget routing;
- measure actual route/classifier overhead on GPU;
- replay held-out requests through the real DreamZero policy;
- require final action cosine >=0.999 and relative L2 <=5%;
- save the worst false-sparse/fallback cases and connect them to downstream
  action/video changes;
- recheck calibration after fixed-shape head-group quantization.
