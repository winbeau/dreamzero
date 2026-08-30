# Dynamic attention Oracle report

Date: 2026-08-30

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

## Exactness and tests

Observer-enabled Dense execution preserves video output, action output, and
returned KV cache exactly in unit tests and the real checkpoint gate. The
latest relevant H200 test groups pass, including real schema aggregation,
condition recovery, task-disjoint M1 table construction, and packed-state
primitives.

## Remaining Oracle work

- perform controlled downstream head/group interventions and record final
  action/video sensitivity rather than inferring it from local output error;
- replay the calibrated M1 policy through the real model and require final
  action cosine >=0.999;
- save worst requests and connect classifier false-sparse events to downstream
  action/video changes.
