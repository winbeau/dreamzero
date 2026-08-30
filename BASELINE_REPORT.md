# DreamZero anchor-sparse baseline report

Date: 2026-08-30

## Status

The released DreamZero-DROID checkpoint runs the embodied anchor-sparse path
end to end with BF16 FlashAttention 2 and the same eight real DiT evaluations
as the dense baseline. A same-hardware control now runs both modes sequentially
on physical GPUs 5--6 and confirms the original speedup, but the result does
**not** meet the paper performance gate.

No result from an OOM-affected run is included in the latency comparison.

## Revisions and environment

- repository: `winbeau/dreamzero`;
- baseline implementation revision: `6292c1335a9d8393416c1012804b51fe5347693f`;
- paired-statistics revision: `8976d92aaf7cc0ba5d80c8696b2441774008d407`;
- checkpoint revision: `96ad344138c66e82536422432ad742f015784942`;
- checkpoint integrity: all ten safetensor shards matched their Hugging Face
  LFS SHA-256 metadata;
- server: H200 NVL. Historical runs used physical GPUs 2--5; after GPU 4 became
  occupied, the user explicitly reassigned the controlled baseline to GPUs
  5--6. GPU 7 was not used;
- environment: repository uv `.venv`, PyTorch BF16, FlashAttention 2, eager
  mode with `DREAMZERO_DISABLE_TORCH_COMPILE=true`;
- inference parallelism: two ranks per server. DreamZero's CFG exchange is a
  two-peer protocol, so dense and sparse are independent two-rank services.

## Matched workload

Both modes use the original DreamZero-DROID policy and checkpoint without
reducing the denoising budget:

- 16 scheduler updates and 8 real DiT evaluations per request;
- 40 transformer layers, hidden width 5120, 40 attention heads;
- 24 action tokens and 1 state token, always computed densely;
- 1760 current video query tokens (two 880-token latent frames);
- 7920 video KV tokens (nine 880-token latent frames);
- three deterministic 180x320 RGB views, robot state, fixed instruction, and a
  persistent session ID;
- 2 warmup requests followed by 12 measured requests, seed `20260830`.

The sparse candidate uses:

- old-frame key keep ratio 0.20 with view-balanced selection;
- the two most recent frames kept dense;
- 2992 / 7920 executed video KV tokens (37.8%);
- current-video keep ratio 0.50;
- five dense prefix layers and five dense suffix layers;
- radius-one spatial delta propagation every five sparse middle layers and on
  the last sparse middle layer;
- one action-conditioned route shared across heads, layers, and all eight DiT
  evaluations in a request.

The initial/cache-rewind conditioning pass and video-only KV-fill pass remain
dense because they do not expose a valid action-conditioned route.

## Exact full-budget control

With historical and current keep ratios set to 1.0, all four ranks reported
exact equality for video output, action output, and the complete updated KV
cache at every transformer layer. Peak allocated memory in the checkpoint DiT
gate was 48.24 GiB per rank.

## Valid end-to-end result: dense 2--3, sparse 4--5

Raw files:

- `e2e_server/20260830/dense_eager_2gpu_long.json`;
- `e2e_server/20260830/sparse_eager_2gpu_long.json`;
- `e2e_server/20260830/paired_comparison_long.json`.

| Metric | Dense | Sparse | Dense / sparse |
| --- | ---: | ---: | ---: |
| Mean | 1.8172 s | 1.5748 s | 1.1539x |
| P50 | 1.7996 s | 1.5953 s | 1.1280x |
| P90 | 1.9554 s | 1.6225 s | 1.2052x |

Paired request statistics:

- geometric-mean speedup: 1.1530x;
- paired bootstrap 95% CI: [1.1295x, 1.1766x], 10,000 resamples;
- sparse faster fraction: 12 / 12 (100%);
- paired speedup range: 1.0922x--1.2217x.

This measurement includes image preprocessing, VAE/image encoding, video KV
updates, the eight real DiT evaluations, scheduler work, action untransform,
and WebSocket transport. It is an open-loop latency test, not a task-quality
result.

## Same-hardware control: dense 5--6, sparse 5--6

To remove inter-GPU-group bias after physical GPU 4 became unavailable, dense
and sparse were run sequentially on the exact same physical pair, GPUs 5--6.
The table uses the hot repeat from each service; both retain the original two
warmup plus twelve measured request protocol and seed.

Raw files:

- `e2e_server/20260830_gpu56/dense_gpu56_repeat.json`;
- `e2e_server/20260830_gpu56_swap/sparse_gpu56_repeat.json`;
- `e2e_server/20260830_gpu56_same_hardware.json`.

| Metric | Dense | Sparse | Dense / sparse |
| --- | ---: | ---: | ---: |
| Mean | 1.8515 s | 1.5963 s | 1.1599x |
| P50 | 1.8360 s | 1.6096 s | 1.1407x |
| P90 | 1.9814 s | 1.6539 s | 1.1980x |

Paired request statistics:

- geometric-mean speedup: 1.1592x;
- paired bootstrap 95% CI: [1.1342x, 1.1832x], 10,000 resamples;
- sparse faster fraction: 12 / 12 (100%);
- paired speedup range: 1.0928x--1.2242x.

The first freshly loaded GPU-5--6 dense run contained three approximately
2.45-second measured outliers and is retained separately rather than selected
as the headline baseline. Using the hot repeat is consistent with steady-state
serving and produces essentially the same conclusion as the original GPU-group
result (1.160x versus 1.154x mean speedup).

## Reverse-orientation audit: dense 4--5, sparse 2--3

Two reverse-orientation dense attempts were invalid. Immediately before each
attempt, GPUs 4--5 contained only the expected DreamZero ranks. During the
first dense warmup request, an unrelated short-lived process allocated about
95 GiB on physical GPU 4, leaving less than 126 MiB free and causing the dense
rank to OOM. The unrelated PIDs were `1647750` and `1659513`; both exited before
their command line could be captured. The affected dense process groups were
terminated by their recorded parent PIDs and their partial outputs are not
used.

Sparse-only reverse-orientation runs on uncontended GPUs 2--3 completed with
means of 1.6361 s and 1.6128 s. They are retained as diagnostic artifacts but
are not paired with dense data and therefore do not satisfy the GPU-swap gate.

A later attempt to swap dense onto GPUs 2--3 was also invalidated when PID
`1691604` began using 25.7 GiB and about 95% utilization on physical GPU 2.
That process was not modified. The same-hardware GPU-5--6 control above is the
authoritative hardware-bias check for this baseline phase.

Artifacts are under
`e2e_server/20260830_gpu_swap/`. Files with `preliminary` semantics or missing
a complete dense pair must not be used in tables or claims.

## Gate decision

| Requirement | Current result | Status |
| --- | ---: | --- |
| Mean end-to-end speedup >= 1.30x | 1.1599x | Fail |
| P50 speedup >= 1.25x | 1.1407x | Fail |
| Paired 95% CI lower bound > 1.15x | 1.1342x | Fail |
| Sparse faster on >= 95% of requests | 100% | Pass |
| Hardware-bias control | same physical GPUs 5--6 | Pass |
| Full-budget output/cache exactness | exact | Pass |

The baseline establishes a real but insufficient end-to-end gain. The next
phase must profile and remove repeated route packing, gather/cat, attention,
MLP/projection, and service overhead before repeating the 20-request pilot and
the full GPU-swapped evaluation.

## Reproduction

The server launch commands and complete sparse configuration are documented in
`docs/RUN_EMBODIED_ANCHOR_SPARSE.md`. Generate paired statistics with:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/compare_dreamzero_server_e2e.py \
  --dense /path/to/dense.json \
  --sparse /path/to/sparse.json \
  --bootstrap-seed 20260830 \
  --output /path/to/paired-comparison.json
```

When a proxy is active, localhost clients must unset proxy variables or set
both `NO_PROXY` and `no_proxy` for `127.0.0.1,localhost`.
