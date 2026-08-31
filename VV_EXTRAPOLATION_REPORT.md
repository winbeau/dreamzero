# Late-step VV extrapolation and online sentinel report

Date: 2026-08-31

## Status

Late-step VV extrapolation is rejected for the current executor. The first online safety module
uses the conditional action-flow sequence as a deployment-observable proxy for
two-step linear predictability. It is implemented and tested, but a
validation-frozen threshold fails the untouched test safety gate. Dense rerun
mode is therefore retained only as a diagnostic implementation and is not
used for the fixed-eight-DiT Sparse Attention result.

Implementation commits:

- `df04445`: online two-step flow sentinel and optional Dense recomputation;
- `d01f9b6`: validation calibration and trace alignment;
- `51b25c8`: robust parsing of interleaved distributed logs;
- `0742329`: frozen-validation threshold evaluation without test retuning.
- `5e6f614`: train/validation/test Dense-VV alpha calibration and sentinel analysis;
- `7b86d88`: executable joint video/action/CFG coverage accounting.

All commits are pushed to `origin/codex/dreamzero-anchor-sparse-opt` and the
H200 checkout is fast-forwarded through them. The focused H200 suite passes 12
tests.

## Runtime contract

For each real DiT evaluation, the module predicts the current conditional
action flow from the previous two real outputs using the actual nonuniform
scheduler spacing:

```text
flow_t ~= flow_(t-1) + alpha * (flow_(t-1) - flow_(t-2))
```

The first two real DiT evaluations are always computed. The sentinel records
prediction cosine and relative L2 from DiT indices 2 through 7. Record-only
mode never changes the model output and every request logs exactly eight DiT
compute steps and eight model calls.

The implementation can recompute a triggered DiT densely, but that path adds
model calls. It is explicitly excluded from the main Sparse Attention result,
whose Dense and Sparse arms must each perform exactly eight real DiT calls.

## Validation calibration

The candidate is the `early2_dense_then_balanced` Packed-M2 table: DiT 0 and 1
are Dense and the remaining six evaluations use the balanced sparse policy.
On 18 task-disjoint DROID validation requests, only one request passes both
final-action gates; 17 fail.

The validation optimizer minimizes triggered DiT recomputations while
requiring every unsafe request to be detected. Its selected rule is:

| Threshold | Value |
| --- | ---: |
| minimum flow cosine | -1.0 (disabled) |
| maximum flow relative L2 | 0.270798 |

Even this fitted rule triggers all 18 validation requests, including the one
safe request. It triggers 24 of 108 checked DiT evaluations, or 1.33 potential
Dense recomputations per request. The sentinel therefore has no request-level
selectivity on validation.

## Frozen-threshold test

The validation thresholds were frozen before evaluating the 18 untouched test
requests. Dense was replayed on GPUs 2--3 and exactly reproduces the prior
Dense actions (`cosine = 1`, relative L2 `= 0`), establishing a stable paired
reference. The record-only sparse arm ran on GPUs 5--6 with eight model calls
per request.

| Metric | Result | Required behavior | Decision |
| --- | ---: | ---: | --- |
| unsafe requests | 15 / 18 | diagnostic | recorded |
| detected unsafe | 14 / 15 | all critical failures | fail |
| false-sparse rate | 6.67% | <1% | fail |
| safe requests falsely triggered | 3 / 3 | low fallback overhead | fail |
| requests triggered | 17 / 18 | selective | fail |
| triggered checked DiTs | 24 / 108 | diagnostic | recorded |
| potential Dense reruns/request | 1.33 | separately timed | too high |

The false-sparse request is
`test_subset032_source022797_early`: final action cosine is 0.993583 and
relative L2 is 11.48%, but its maximum flow-prediction relative L2 is only
0.266469 and remains below the validation threshold. This counterexample
rejects the current linear-flow residual as a safety certificate.

The sparse table itself measures 1.381x paired geometric-mean end-to-end
speedup against the contemporaneous Dense replay, with 95% bootstrap interval
[1.363x, 1.398x] and all 18 sparse requests faster. It is rejected on quality:
mean/minimum action cosine are 0.993664/0.967604 and mean/maximum relative L2
are 10.04%/25.39%.

## Reproducibility note

Dense actions are exactly stable across repeated service replay. The same
sparse table shows small cross-restart output variation, so record-only
identity is asserted only within the same loaded service instance. The
validation record-only run is exactly identical to its no-sentinel baseline;
the test quality comparison uses a freshly replayed, exactly stable Dense
reference and treats sparse restart variation as part of deployment risk.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/e2e/20260830_droid_108_round1/
    flow_sentinel_record_validation18/
    flow_sentinel_record_test18/
    dense_replay_test18_20260831/
```

## Decision and next gate

- reject action-flow linear residual as the sole confidence/fallback signal;
- do not spend extra Dense model calls on its rerun path;
- keep actual VV extrapolation separate from Sparse Attention timing;
- before enabling VV extrapolation, calibrate per-head/per-layer VV residuals
  only on `predictable-late` routes and require a validation-frozen sentinel
  with false-sparse below 1%;
- preserve a fixed-eight-call future-step promotion path if a predictive
  signal is found, rather than hiding recomputation inside the main speedup.

## Dense-Oracle VV extrapolation gate

The schema-3 Dense Oracle already stores the real VV output signature for all
108 requests, eight DiT evaluations, 40 layers, 40 heads, conditional and
unconditional CFG branches, and video/action query kinds. This permits an
actual VV extrapolation test without another model run.

Alpha is fitted only on the 72-request train split, independently for each
`(dit, layer, branch, modality, head)`, then clipped to `[0, 2]`. Eligibility
is frozen by requiring both train and validation p05 cosine >=0.999 and p95
relative L2 <=5% in the proposed late region (DiT 5--7, layers 28--39). Test
contains 18 untouched requests.

The fitted dynamics do not follow scheduler spacing. Mean fitted alpha by DiT
2--7 is `[0.326, 0.509, 0.130, 0.170, 0.157, 0.522]`, while the nonuniform
scheduler formula gives `[1.077, 2.000, 1.788, 1.497, 0.556, 1.403]`. Across
all late-region test signatures, fitted alpha is best among reuse, alpha=1,
and scheduler alpha, but only 4.04% of signatures pass the local quality gate;
mean cosine/L2 are 0.9803/17.07%.

Validation freezes only 45 of 5,760 late signature cells (0.781%). On test,
those selected signatures have 0.864% false extrapolation, mean cosine
0.999862, and mean relative L2 1.481%. A previous-residual sentinel calibrated
at relative L2 0.09933 reduces test false extrapolation to 0.745%, but still
leaves a worst relative L2 of 5.54%.

The decisive systems gate is joint executability:

| Unit required to skip compute | Eligible | Fraction |
| --- | ---: | ---: |
| one modality/branch/head signature | 45 / 5,760 | 0.781% |
| one branch/head with both video and action safe | 1 / 2,880 | 0.0347% |
| one head with both modalities and both CFG branches safe | 0 / 1,440 | 0% |

No full Head Attention unit can therefore be skipped while satisfying the
frozen quality rule. Caching two full VV histories and adding extrapolation
buffers would create memory and control overhead with zero executable compute
saving. Runtime VV extrapolation is not implemented or included in Sparse
Attention timing. The offline analyzer and alpha table are retained as the
required negative ablation.

Artifacts:

```text
dynamic_m1_m2/vv_extrapolation/
  20260831_dense_oracle_alpha_v1/
  20260831_dense_oracle_alpha_v2/
```
