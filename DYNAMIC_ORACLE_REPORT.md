# Dynamic attention Oracle report

Date: 2026-08-30

## Status

The dense Oracle capture path and a real DreamZero-14B smoke gate are complete.
Multi-task collection over all eight real DiT evaluations and all 40 layers is
not complete.  The current three-layer/two-query result validates the pipeline
only; it is insufficient evidence for any timestep, layer, or head sparsity
law.

Implementation revisions:

- `04f7d73210b046c06bdf78b229506494e465a105`: chunked Dense Oracle capture,
  runtime wiring, server flags, and output-invariance tests;
- `c8e5e64`: bounded layer/query capture and real-checkpoint smoke gate.

## What is collected

For both video-query-to-video-key and action-query-to-video-key attention, the
collector records each `(request, scheduler step, real DiT index, layer,
head)` at keep ratios 100%, 75%, 50%, 35%, 25%, 20%, and 10%:

- Dense attention mass retention: mean, p05, and minimum;
- top-p token counts for p=0.50/0.75/0.90/0.95;
- exact top-k-renormalized attention-output cosine and relative L2;
- normalized entropy and maximum attention mass;
- nested ranked key profiles;
- support turnover across real DiT evaluations;
- Qa-to-Kv versus sampled Qv-to-Kv key-importance correlation;
- a conservative minimum local Oracle budget using p05 mass >=0.90,
  p05 output cosine >=0.999, and p95 output relative L2 <=5%.

Final action sensitivity is deliberately not inferred from local attention
error.  It requires downstream activation intervention and remains a separate
pending Oracle gate.

## Memory-safe implementation

A full `40 x 1785 x 7920` score tensor has more than half a billion elements
per layer.  The collector therefore:

1. selects deterministic, spatially distributed video queries or every query;
2. computes exact FP32 QK/softmax in configurable query chunks;
3. scans all required keep-ratio buckets from one descending ranking;
4. writes only per-head statistics and nested aggregate key rankings;
5. stores artifacts outside Git.

Oracle mode is rejected when anchor sparsity is enabled.  It is analysis-only
and its latency is never included in Dense or Sparse performance results.

## Unit and integration gates

H200 CPU test command:

```bash
CUDA_VISIBLE_DEVICES= uv run --frozen --no-sync python -m pytest -q \
  tests/test_dynamic_attention_oracle.py \
  tests/test_embodied_anchor_sparse.py \
  tests/test_embodied_anchor_attention_integration.py
```

Result: 28/28 passed.  Coverage includes:

- deterministic query sampling;
- nested profile construction;
- full-budget local attention exactness;
- tail-aware Oracle budget selection and Dense fallback;
- head-importance correlation and support turnover;
- request/step/layer artifact writing;
- layer filtering;
- Oracle observation without changing self-attention output or KV cache.

## Real-checkpoint smoke gate

Physical GPU 2 loaded the released 16.48B checkpoint and compared the same
Dense forward with Oracle disabled and enabled.  The bounded smoke captured
layers 0, 20, and 39, two video queries, two action queries, all 40 heads, and
all seven keep-ratio buckets.

| Gate | Result |
| --- | ---: |
| Requested/recorded layers | 3 / 3 |
| Head coverage | 40 / 40 |
| JSON layer records | 3 |
| Nested support profiles | 6 |
| Video output exact | yes |
| Action output exact | yes |
| Returned cache exact | yes |
| Peak allocated memory | 36.41 GiB |
| Artifact size | 3.1 MB |

Raw artifacts:

`dynamic_m1_m2/oracle_smoke/20260830_gpu2/`

Reproduction:

```bash
CUDA_VISIBLE_DEVICES=2 DREAMZERO_DISABLE_TORCH_COMPILE=true \
  uv run --frozen --no-sync python \
  benchmarks/validate_dynamic_attention_oracle.py \
  --model-path /data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse/checkpoints/DreamZero-DROID \
  --output-dir /data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/dynamic_m1_m2/oracle_smoke/20260830_gpu2 \
  --physical-gpu 2 --layer-indices 0 20 39 \
  --max-video-queries 2 --max-action-queries 2 --query-chunk-size 1
```

## Smoke observations, not claims

At 20% keep ratio, mean retained mass over the 40 heads is non-monotonic across
layers 0/20/39 for both video and action queries.  The late layer is not
uniformly safer to sparsify in this tiny sample.  This contradicts using layer
depth alone as a hard-coded rule and supports the planned task-disjoint Oracle
collection and confidence fallback.  Because only two queries and one
synthetic input were used, these numbers must not appear as paper evidence.

## Remaining Oracle requirements

- collect all eight real DiT evaluations and all 40 layers on multiple real
  tasks, instructions, trajectory stages, and both CFG ranks;
- expand video-query coverage and quantify sampling convergence;
- add downstream per-head/group interventions for final action/video
  sensitivity;
- compute task variance, high-frequency change, and VV temporal features;
- generate complete timestep-layer-head heatmaps;
- derive train/validation/test Oracle labels without task leakage.
