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
raised to 87.5%. A small number of M1-controlled head groups, confidence Dense
fallback, and a full eight-DiT policy replay are therefore required before this
phase can pass its quality gate.

Implementation commits:

- `28bbf47`: fixed-shape 8-by-40 dynamic budget table and Packed M2 runtime;
- `6fe77ec`: reproducible Oracle-ordered fixed/timestep/layer/joint ablations;
- `aa9706f`: per-rank early/late checkpoint gate and actual token accounting;
- `e7d8c2d`: align the checkpoint gate with the eight real diffusion timesteps.

All listed commits are pushed to
`origin/codex/dreamzero-anchor-sparse-opt`, and the H200 checkout is
fast-forwarded through `e7d8c2d`.

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

## Interpretation and next gate

The timestep hypothesis is useful for compute allocation: the Oracle-supported
late budget produces a substantially smaller packed shape and a larger speedup.
The layer hypothesis must remain non-monotonic because late layers recover
strongly in the Dense Oracle.

However, one shared token budget across every head is not a viable final
policy. Raising all heads together spends nearly Dense compute on early DiT
without recovering video. The next executor revision must therefore:

1. map the calibrated M1 output to a small number of fixed head groups;
2. keep critical and confidence-uncertain groups Dense;
3. use group-specific nested Q/K/V lengths with fixed-shape FA2 calls;
4. apply the O projection without dropping the 25 Dense registers;
5. log category, confidence, fallback, and effective group budget;
6. replay all eight real DiT evaluations on held-out DROID requests before any
   final quality claim.

The aggressive and quality tables remain performance/structure ablations. No
current dynamic table is promoted as the final M1 policy.
