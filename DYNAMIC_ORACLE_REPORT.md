# Dynamic attention Oracle report

Date: 2026-08-31

## Status

The full local-attention Oracle collection is complete on 108 real DROID
requests. It covers every requested real DiT evaluation, Transformer layer,
head, query kind, and CFG branch. The data supports a monotonic timestep law
but rejects a monotonic layer-depth law.

One required Oracle item remains open: downstream intervention that attributes
final action/video sensitivity to each head or a controlled head group. Local
attention-output error is not reported as final action sensitivity. Therefore
the overall research Goal and the complete Oracle phase remain active.

Phase implementation commits:

- `04f7d73`: memory-bounded Dense attention Oracle observer;
- `a3f7b4f`: conditional/unconditional CFG evidence separation;
- `58cb91a`: temporal VV signatures and adjacent-DiT change features;
- `e95aee6`: resumable real DROID collection runner;
- `9cd8c4f`, `3837f84`: immutable task-disjoint DROID subset and balanced quotas;
- `b113144`: video-query sampling-convergence analysis;
- `329de40`, `839c164`: full dataset aggregation and compact M1 table;
- `e0adbf3`, `9f5c60b`: audited condition sidecar recovery;
- `dc7fb2c`, `fa9ed4d`: reproducible paper-statistics summary.
- `7827c5c`: research-gated final-video latent return and paired downstream
  action/video sensitivity recording.
- `3da14e9`: applied-count traces and a strict scale-one history-snapshot
  action/video exactness gate.

All listed commits are pushed to `origin/codex/dreamzero-anchor-sparse-opt` and
the H200 checkout was fast-forwarded through them.

## Protocol

The released DreamZero-14B AR model uses 16 scheduler steps. Both Dense and
Sparse main-result protocols retain the same eight real DiT evaluations at
scheduler indices `[0, 1, 2, 6, 10, 13, 14, 15]`; this Oracle observes all
eight and never changes model outputs.

The immutable DROID subset contains 36 source episodes and 108 requests:

| Split | Episodes | Requests |
| --- | ---: | ---: |
| train | 24 | 72 |
| validation | 6 | 18 |
| test | 6 | 18 |

Every episode contributes early, middle, and late trajectory requests. The
three stages use instruction slots 0, 1, and 2, respectively. Exact-normalized
instructions have no connected-component overlap between splits.

Each request builds seven real historical latent frames through three AR
warmup blocks and captures two current frames. Thus every attention record has
7,920 video keys. The observer records:

- all 8 real DiT evaluations;
- all 40 layers and 40 heads;
- video-query to video-key attention using 32 deterministic spatial queries;
- all 24 action queries against video keys;
- conditional and unconditional CFG branches separately;
- keep ratios 100%, 75%, 50%, 35%, 25%, 20%, and 10%;
- Dense mass retention, top-p token counts, output cosine/relative L2,
  entropy, maximum attention mass, support turnover, Qa/Qv importance
  correlation, and temporal VV output change.

The conservative compact label for `(request,t,l,h)` is the maximum minimum
budget across video/action queries and conditional/unconditional branches.
Each local Oracle budget must satisfy p05 mass >=0.90, p05 output cosine
>=0.999, and p95 output relative L2 <=5%.

## Capture and aggregation gates

Physical GPUs 2 and 3 collected independent 54-request shards. All artifacts
are outside Git.

| Gate | Result |
| --- | ---: |
| Planned/passed requests | 108 / 108 |
| Unique request keys | 108 |
| Missing/extra manifest keys | 0 / 0 |
| Records per request | 640 |
| Total layer records | 69,120 |
| Full head-table rows | 5,529,600 |
| Compact M1 rows | 1,382,400 |
| Peak allocated memory | 61.616--61.618 GiB |
| Capture time per request | 27.25--28.89 s |
| Full aggregation gate | passed |

The main capture occupies about 70 GiB. The final aggregated directory is
2.2 GiB, including a 1.996 GiB complete head table and a 358 MiB compact M1
table.

Main artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/oracle_main/20260830_q32_schema3_cond/
  dynamic_m1_m2/oracle_analysis/20260830_q32_schema3_cond_full/
```

The initial runner accidentally omitted five request-level state/action
condition values when `warmup_history_blocks>0`. Raw attention evidence was
unaffected. A read-only sidecar reconstructed the exact released evaluation
modalities from the compact DROID parquet files. All 108 values are finite;
the original capture JSONL, profiles, and predictions were not rewritten.

## Query sampling convergence

Eight or sixteen sampled video queries are not adequate substitutes for the
32-query reference:

| Candidate | Oracle-label agreement | False-sparse vs q32 | Over-conservative |
| --- | ---: | ---: | ---: |
| q8 | 64.07% | 19.22% | 16.71% |
| q16 | 71.23% | 11.58% | 17.19% |

All M1 evidence therefore uses q32. Action-query statistics use every action
query and are identical across these convergence pilots.

## Oracle budget distribution

Across 1,382,400 conservative `(request,t,l,h)` labels:

| Keep ratio | Fraction |
| --- | ---: |
| 10% | 2.93% |
| 20% | 4.40% |
| 25% | 2.85% |
| 35% | 7.60% |
| 50% | 15.88% |
| 75% | 39.66% |
| Dense | 26.68% |

The mean conservative Oracle budget is 68.91%. A fixed 20% route preserves
worst-branch/query p05 mass >=0.90 for only 13.69% of head states. This
directly rules out fixed 80% sparsity as a quality-safe default under the
current local thresholds.

## Timestep law: supported

The global conservative budget falls at every adjacent real DiT evaluation,
from 70.23% at DiT 0 to 67.01% at DiT 7. Separate video and action budgets are
also monotonic across all eight evaluations.

Episode-level 200-repeat bootstrap, early minus late timestep:

| Query | Mean | 95% CI | Positive repeats |
| --- | ---: | ---: | ---: |
| video | +4.51 pp | [+4.29, +4.70] pp | 100% |
| action | +3.28 pp | [+3.18, +3.38] pp | 100% |

The fixed three timestep buckets have conservative mean budgets of 70.06%
(DiT 0--2), 69.24% (DiT 3--4), and 67.54% (DiT 5--7). The effect is real but
smaller than the layer/head effect, so timestep alone cannot provide the
target speedup.

## Layer law: monotonic hypothesis rejected

Layer sensitivity is U-shaped, not monotonically decreasing. The middle
trough is sparse-friendly, while the late stack recovers sharply:

| Layer bucket | Mean Oracle budget | Dense rate |
| --- | ---: | ---: |
| 0--11 | 67.01% | 22.94% |
| 12--27 | 65.16% | 13.48% |
| 28--39 | 75.82% | 48.01% |

Layer 39 has a 90.35% mean conservative budget and a 75.49% Dense rate. Layer
0 is also expensive (81.12%, 51.35% Dense), whereas layers 12--18 contain much
of the trough.

Episode-level bootstrap, early third minus late third:

| Query | Mean | 95% CI | Positive repeats |
| --- | ---: | ---: | ---: |
| video | -13.81 pp | [-13.94, -13.66] pp | 0% |
| action | -6.77 pp | [-6.96, -6.58] pp | 0% |

There are 21 video and 17 action adjacent-layer monotonicity violations.
Dynamic M2 must therefore use an early/middle/late-recovery structure and test
whether the final output layer should remain Dense.

## Head, task, CFG, and trajectory evidence

Head identity is informative but not sufficient for a permanent class. Mean
budgets range from 56.41% (head 28) to 77.94% (head 14), yet every head reaches
Dense at its 90th percentile. This supports dynamic head classification with
confidence fallback, not static aggressive-head pruning.

Cross-request budget standard deviation averages 4.82 percentage points and
has a 95th percentile of 12.24 points. The mean conditional/unconditional
budget gap is 0.41 points and its 95th percentile is 1.62 points; keeping CFG
branches separate was still necessary to avoid false-sparse labels.

Trajectory-stage mean budgets are close: early 69.19%, middle 68.81%, late
68.74%. Stage alone is therefore a weak router feature compared with
timestep/layer/head and temporal dynamics.

## Same-noise downstream final-action intervention pilot

Commits `2d0e8f5` and `232ec62` add a Dense-only intervention immediately
before the attention output projection and a same-process DROID paired runner.
The runner executes the complete Dense history and target, resets the policy,
then repeats the identical history with an intervention only on the second
target. Both trajectories use the released fixed action/video noise seed and
the same physical GPUs. Request-level overrides are disabled unless the server
is launched with the explicit research flag.

A scale-one control at `(DiT 0, layer 39, head 14)` is exactly equal after the
complete history and eight real DiT evaluations:

| Control gate | Result |
| --- | ---: |
| final action cosine | 1.000000 |
| final action relative L2 | 0% |
| final action maximum absolute difference | 0 |
| target intervention count | 1 |

This same-process control was necessary. Comparing independently launched
services produced a nonzero difference despite a scale-one tensor no-op,
because cross-process/GPU execution is not a sufficiently strict causal
control for this measurement.

The first scale-zero pilot uses one untouched test episode at its early,
middle, and late task stages. Each row removes one attention head only once in
the full denoising trajectory:

| Intervention | Mean action rel-L2 | Max action rel-L2 | Min cosine |
| --- | ---: | ---: | ---: |
| DiT 0 / layer 39 / head 14 / all queries | 1.693% | 3.154% | 0.999516 |
| DiT 7 / layer 39 / head 14 / all queries | 0.061% | 0.069% | 1.000000 |
| DiT 0 / layer 0 / head 14 / all queries | 0.652% | 1.164% | 0.999933 |
| DiT 0 / layer 20 / head 14 / all queries | 2.137% | 3.146% | 0.999524 |
| DiT 0 / layer 39 / head 28 / all queries | 1.661% | 2.530% | 0.999717 |
| DiT 0 / layer 39 / head 14 / registers only | 1.940% | 2.924% | 0.999587 |
| DiT 0 / layer 39 / head 14 / video queries only | 1.212% | 2.396% | 0.999734 |

Four conclusions are already supported, although this is not yet the complete
Oracle scan:

1. The first real DiT evaluation is about 27.7x more action-sensitive than the
   last for the same layer/head under mean relative L2.
2. Layer sensitivity is non-monotonic: layer 20 exceeds both layer 0 and layer
   39 in this controlled final-action measurement.
3. The local-budget head ordering is not a downstream-action ordering. Head 14
   has the highest mean local Oracle budget and head 28 the lowest, but their
   controlled action errors at DiT 0/layer 39 are similar.
4. Query type interacts with trajectory stage. Register-query removal is
   strongest at the early stage, while video-query removal is strongest at the
   late stage; neither path can be discarded globally.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/downstream_oracle/20260831_same_process_pair/
```

### Task-disjoint timestep validation

The early/late DiT contrast was then expanded to all 18 validation requests
(six source episodes, each with early/middle/late stages). Every intervention
still removes only head 14 once at layer 39, and every paired trajectory keeps
the complete three-block history and eight real DiT evaluations.

| Real DiT | Mean action rel-L2 | Max action rel-L2 | Min cosine |
| ---: | ---: | ---: | ---: |
| 0 | 1.268% | 4.518% | 0.999235 |
| 7 | 0.0427% | 0.0662% | 0.9999998 |

The mean downstream sensitivity differs by 29.7x. For DiT 0, early/middle/late
mean relative L2 is 1.996%/0.438%/1.371%; for DiT 7 all three stage means are
below 0.047%. The worst DiT-0 request is
`validation_subset024_source018470_early` ("Move the pineapple plushy
backwards"), with cosine 0.999235, relative L2 4.518%, and maximum absolute
action difference 0.16815. Thus a single false-sparse critical-head decision
can violate the required action-cosine gate even though its average error is
much smaller.

Commits `8975854`, `185a642`, and `c144652` add a tested resumable grid runner
and the first deployment-shaped four-head shared-group scan configuration.
The grid writes every request/candidate row incrementally before final
aggregation, so an interrupted long Oracle run can resume without discarding
completed trajectories.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/downstream_oracle/20260831_validation18/
```

### Final-video downstream metric path

Commit `7827c5c` closes the instrumentation gap between local attention-output
error and final generated-video error. When and only when the server is
launched with the downstream research override enabled, a target request may
return both the normal action and the final `video_pred` latent. The normal
service response remains action-only.

The single-candidate and resumable grid runners can now record paired final
video cosine, relative L2, maximum absolute difference, latent shape, stage
means, and the worst request while retaining the existing action metrics. Raw
video latents are not written to JSON, avoiding artifact inflation. Resume
rejects action-only/video-metric schema mixing. With Dense-history snapshots,
one baseline latent is compared with every candidate restored from the same
pre-target state.

Commit `3da14e9` also returns the exact intervention trace with research-only
video responses. Every measured candidate must match its requested DiT,
layer, heads, scale, CFG branch, and query scope and must report exactly one
application; the Dense baseline must report zero. The checked-in
`d0_l39_h12_15_scale1_exactness.json` candidate and
`validate_downstream_exactness.py` then fail unless the grid used a restored
Dense-history snapshot, used scale one, preserved the final action array
elementwise, and produced zero final-video relative L2 and maximum absolute
difference. A cosine merely close to one cannot hide a nonzero difference.

Local gates for these changes are 23 passing benchmark/client tests, Python
compilation, and `git diff --check`. The GPU-dependent wrapper tests and the
real-checkpoint strict gate remain pending because both configured H200 SSH
routes were unavailable at commit time.

## Exactness and tests

Observer-enabled Dense execution preserves video output, action output, and
returned KV cache exactly in unit tests and the real checkpoint gate. The
latest relevant H200 test groups pass, including real schema aggregation,
condition recovery, task-disjoint M1 table construction, and packed-state
primitives. This statement predates the new WebSocket video-return path; that
path is not considered exact until its pending H200 gate passes.

## Remaining Oracle work

- expand the controlled downstream intervention from this pilot to a
  task-disjoint `(timestep, layer, shared-head-group, query-type)` scan and
  collect the now-instrumented final-video sensitivity;
- replay the calibrated M1 policy through the real model and require final
  action cosine >=0.999;
- save worst requests and connect classifier false-sparse events to downstream
  action/video changes.
