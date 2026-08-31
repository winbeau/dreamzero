# DreamZero dynamic M1/M2 final status

Date: 2026-08-31

## Decision

The current `shared timestep/layer table + maximum-Head promotion +
request-wide fallback gate` line is closed as a negative result. It does not
meet the required quality-safe end-to-end speedup of `>=1.35x`.

This decision does not invalidate sparse WAM attention or Packed M2. It
separates three different claims that must not be conflated:

| Level | Trustworthy result | Quality status | Decision |
| --- | --- | --- | --- |
| Packed DiT executor ceiling | `2.42x--2.82x` single-DiT at fixed 20% | large video error | kernel result only |
| Stable static end-to-end table | `1.0738x--1.0908x` by split, 108/108 faster | 92/108 action-safe | reject |
| Quality-gated end-to-end route | `1.0094x` validation, `1.0137x` development test | held-out mixed actions pass, train episode-CV has 6/72 false-sparse | reject safety and speed |

There is therefore no accepted quality-safe acceleration Claim from the
current route. The three-request `1.313x` pilot is not used: a Dense outlier
inflated it, while validation18 gives the stable `1.0738x` estimate.

## Completed contributions

1. **Real DreamZero Oracle evidence.** The collection covers 108 DROID
   requests, 36 source episodes, all eight real DiT evaluations, 40 layers,
   40 Heads, both CFG branches, 32 deterministic video queries, all 24 action
   queries, and seven keep ratios. It contains 69,120 layer records,
   5,529,600 full Head rows, and 1,382,400 compact M1 rows.
2. **Measured dynamic laws.** The conservative budget decreases from 70.23%
   at DiT 0 to 67.01% at DiT 7, but the effect is modest. Layer sensitivity is
   U-shaped rather than monotonically sparse: layers 28--39 average 75.82%
   keep and layer 39 averages 90.35%. Every Head is Dense at its 90th
   percentile, so permanent aggressive-Head pruning is unsupported.
3. **Deployment-shaped M1.** Packed-proxy-v2 uses only causal runtime
   features, compares GMM, logistic regression, Gradient Boosting, a small
   MLP, and a cost-sensitive classifier, and implements confidence promotion
   and Dense fallback. The selected local classifier has 0.720% test
   false-sparse and retains at least 0.9 Dense mass on 99.9787% of held-out
   rows, but local attention labels do not certify final-action safety.
4. **Dynamic nested budgets.** Timestep/layer tables, separate history/current
   budgets, propagation-aligned five-layer segments, nested anchor prefixes,
   and full-budget Dense fallback are implemented. Budget changes preserve
   the fixed 16 scheduler steps and eight real DiT calls.
5. **Packed M2.** The middle stack gathers once, keeps current activations and
   timestep modulation packed, runs Q/K/V/O, self-attention, cross-attention,
   and FFN on packed video tokens plus all 25 action/state registers, reuses
   routes and historical K/V, preserves original RoPE positions, and scatters
   only at propagation or Dense recovery boundaries. Full-budget video,
   action, and returned K/V are exact against Dense.
6. **Complete static frontier evaluation.** The late4/S4/H50Q50 table was run
   on 72 train, 18 validation, and 18 development-test requests with three
   real history calls and eight real DiTs per target. All logs and artifacts
   are saved outside Git.
7. **Safety mechanisms evaluated.** Maximum-Head promotion, a causal
   request-wide promotion gate, action-flow sentinel, Dense-action history,
   Dense suffix recovery, propagation changes, maximum-current readout, and
   offline VV extrapolation were implemented or analyzed and rejected where
   their measured gates failed.

## Stable 108-request result

| Split | Requests | Dense mean | Sparse mean | Paired geomean | CI95 | Faster | Action-safe |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| train | 72 | 1.8799 s | 1.7236 s | 1.0908x | [1.0874x, 1.0941x] | 72/72 | 61/72 |
| validation | 18 | 1.9034 s | 1.7725 s | 1.0738x | [1.0691x, 1.0786x] | 18/18 | 14/18 |
| development test | 18 | 1.8985 s | 1.7471 s | 1.0867x | [1.0798x, 1.0938x] | 18/18 | 17/18 |

Aggregate action safety is 92/108. The 16 failures include early, middle, and
late trajectory stages; stage identity is not a sufficient fallback rule.
The worst train/test relative L2 values are 8.39% and 8.53%, respectively.
This static row has not undergone the required three-round GPU exchange and
must not be presented as a main performance result.

The test split was inspected during method development. It is retained as
development evidence, not as a new locked final holdout. A future route needs
a newly frozen holdout before a paper Claim.

## M1 outcome

The per-Head local classifier passes its statistical local-attention gates but
cannot safely drive the current shared executor:

- maximum-Head promotion finds 6--32 fallback Heads in every held-out
  candidate cell and promotes 100% of cells to Dense;
- a request-wide cost-sensitive Gradient Boosting gate retains the sparse
  table for only 3/18 validation and 3/18 development-test requests;
- its mixed speedups are 1.0094x and 1.0137x, with only 16.7% of requests
  strictly faster;
- train leave-one-source-episode-out replay still has 6/72 false-sparse
  requests across four folds.

The gate artifact correctly records `passed: false`. It was not connected to
the live inference service and supports no M1 acceleration Claim.

## M2 outcome

Packed M2 proves that removing middle-stack token work can exceed the DiT
target: fixed 20% reaches 2.821x with one Dense boundary layer and 2.420x with
three. The corresponding video relative L2 is 224.47% and 127.25%, so this is
a systems ceiling rather than a usable policy.

The main executor bottleneck is now policy geometry, not proof that packing
works. Heterogeneous per-Head groups pay projection, packing, and FA2 launch
overhead while the shared activation/FFN stays near the largest active shape.
The fully coupled online M1 pilot is slower than Dense at 0.841x.

## Paper-usable negative results

The complete measured ablation table is in `ABLATION_REPORT.md`. The central
negative findings are:

- **Local sparsity is not downstream sensitivity.** High local attention mass
  or output fidelity does not reliably predict final action error.
- **Timestep is monotonic but weak.** Later DiTs are modestly more sparse in
  aggregate, yet adding DiT index 3 to the static frontier raises worst action
  L2 to 18.41%.
- **Layer depth is not monotonic.** The Oracle is U-shaped; adding the fifth
  segment (layers 21--25) fails the action gate.
- **Current-query compression is quality-critical.** Full-segment Q35 reaches
  8.72% worst action L2, and an individual Q25 cell reaches 6.37%; Q50 is the
  measured floor for the current shared segment family.
- **Per-Head sparsity can lose at the kernel level.** Head-as-batch varlen FA2
  takes 2.11 ms versus 1.54 ms for regular fixed-shape FA2; two sliced groups
  take 2.36 ms. The coupled dynamic Head route reaches only 0.841x end to end.
- **Cosmetic output recovery is insufficient.** A larger propagation radius
  improves checkpoint video L2 to 4.233%, but validation18 has zero safe
  actions and 9.553% mean action L2. Dense suffixes similarly improve video
  without repairing accumulated action-state error.
- **Stale token visibility is not recovery.** Maximum-prefix action readout at
  all packed layers worsens action L2 from 1.849% to 2.147%.
- **Coarse profile/request selection has a low ceiling.** Even a measured
  Oracle over balanced/conservative/Dense profiles reaches at most 1.1254x;
  the causal request gate realizes about 1.01x.
- **The current sentinel is not selective.** The action-flow sentinel misses
  one of 15 unsafe development-test requests and triggers every safe request.
- **VV extrapolation has no executable unit.** Only 0.781% of late signatures
  are locally eligible, and zero complete Heads are safe across both
  modalities and CFG branches, yielding no skippable Attention unit.

## Why the current line fails

The table is too coarse to isolate the few risky state/head regions, while the
safe fallback decisions are too broad:

1. a single fallback Head forces a shared cell Dense;
2. avoiding that with 3--4 heterogeneous Head groups loses the Packed kernel
   advantage;
3. falling back an entire request preserves held-out quality only by removing
   almost all sparse coverage;
4. expanding shared sparse coverage in either timestep or layer direction
   crosses the final-action gate before reaching the E2E target.

Consequently the route fails both required conditions: quality-controlled
routing and `>=1.35x` end-to-end speedup.

## Next route, not started

The next experiment should replace both extremes with a small, fixed-shape
**risk-region executor**:

1. obtain downstream action-sensitivity labels for fixed
   `(propagation segment, shared Head class)` regions rather than individual
   Heads or whole requests;
2. freeze two or at most three Head memberships over a propagation segment,
   with a critical group Dense/high-budget and a normal group using nested
   H/Q prefixes;
3. select region budgets from causal Packed features, but keep group shapes
   fixed so FA2 and projection work remain amortized;
4. update hidden states for promoted tokens at segment boundaries instead of
   exposing stale maximum-prefix state;
5. train and calibrate only on train/validation, then evaluate on a newly
   locked holdout with GPU exchange and closed-loop non-inferiority.

This route targets the missing granularity between request-wide fallback and
unprofitable per-Head kernels. No experiment for it is launched in this
closure phase.

## Outstanding paper gates

- no accepted policy reaches `>=1.35x` quality-safe E2E speedup;
- no accepted policy has three GPU-exchange rounds;
- generated-video quality is incomplete for the final gate;
- closed-loop success and the five-point non-inferiority test are incomplete;
- the existing test split is development evidence and must be replaced by a
  newly locked holdout for the next route.

The overall research program is therefore not marked complete. The current
shared-table/request-gate branch is, however, fully measured and closed.

## Reproducibility

Primary artifact roots:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/dynamic_m1_m2/
  oracle_main/20260830_q32_schema3_cond/
  oracle_analysis/20260830_q32_schema3_cond_full/
  m1_proxy/20260831_proxy_v2_full_287f8a8/
  m1_classifier/20260831_packed_proxy_v2_07e0a58/
  dynamic_budgets/20260831_guarded_segments_late4_s4_h50q50_max/
  e2e/20260831_guarded_segments_late4_s4_h50q50_train72/
  e2e/20260831_guarded_segments_late4_s4_h50q50_validation18/
  e2e/20260831_guarded_segments_late4_s4_h50q50_test18/
  request_gate/20260831_shared_h50q50_proxy_v2_episode_cv/
  vv_extrapolation/20260831_dense_oracle_alpha_v2/
```

All benchmark clients completed and their logs are retained in the artifact
directories. Resident Dense/Sparse service processes are idle model holders,
not active experiments. The last pre-closure repository synchronization point
was clean at `2bc2643` on local, origin, and H200.
