# DreamZero anchor-sparse profile report

Date: 2026-08-30

## Executive result

Real-checkpoint profiling confirms that routing is not the bottleneck. The
current candidate reduces FlashAttention substantially, but a sparse DiT still
spends about 51.7 ms in `addmm` and 21.2 ms in `cat` per forward. Cached-route
`gather` costs about 3.0 ms, while constructing a cold action-conditioned route
adds less than 1 ms.

At server level, 240.8 ms of the 249.2 ms mean model-stage saving comes from
diffusion. Text/image encoding, VAE, and video KV creation are effectively
unchanged, as intended by the fair baseline.

## Setup and artifacts

- code revision with profiler: `7a882d57397b5eec1d0dae7abdac4e09cb33c568`;
- log summarizer revision: `bbb5dc5716828b40b6cc25b26fde8b332991c479`;
- checkpoint: DreamZero-DROID revision
  `96ad344138c66e82536422432ad742f015784942`;
- physical GPUs: H200 NVL 5--6;
- dtype/backend: BF16 FlashAttention 2;
- geometry: seven cached frames, two current frames, 24 action tokens, one
  state token, 40 layers;
- candidate: historical keep 0.20, current keep 0.50, recent-dense frames 2,
  dense prefix/suffix 5/5, propagation radius 1 every 5 middle layers, denoise
  route reuse enabled;
- timing: two warmup forwards and four measured forwards per GPU;
- traces: `profiles/checkpoint_ops/20260830_gpu56/`;
- shape-grouped trace: `profiles/checkpoint_ops/20260830_gpu5_shapes/`;
- server-stage summaries: `profiles/server_stages/20260830_gpu56/`.

Chrome traces are 12--14 MB each and remain in the external artifact root; no
large trace is committed to Git.

## Checkpoint DiT gate

| GPU | Dense p50 | Sparse p50 | Speedup |
| ---: | ---: | ---: | ---: |
| 5 | 197.48 ms | 160.08 ms | 1.234x |
| 6 | 194.38 ms | 155.98 ms | 1.246x |

A separate shape-grouped run on GPU 5 measured 198.62 ms dense and 159.16 ms
sparse (1.248x), confirming that shape recording did not change the conclusion.

The full-budget control remained exactly equal for video output, action output,
and every layer's updated KV cache. Peak allocated memory was 48.23 GiB.

The 0.20/0.50 candidate has synthetic-input relative L2 of 57.32% for video and
5.55% for action. These values are sensitivity diagnostics, not task quality;
the action value is slightly above the 5% numerical target and must be guarded
in later configuration search.

## CUDA operator breakdown

The table uses the GPU-5 trace and reports self CUDA time for one DiT forward.
Profiler trace time is used for attribution; the CUDA-event measurements above
remain the authoritative latency numbers.

| Operator family | Dense | Sparse cached route | Change |
| --- | ---: | ---: | ---: |
| `aten::addmm` | 63.62 ms | 51.72 ms | -11.90 ms |
| FlashAttention 2 forward | 44.51 ms | 19.27 ms | -25.24 ms |
| `aten::cat` | 26.07 ms | 21.19 ms | -4.88 ms |
| `aten::gather` | negligible | 2.97 ms | +2.97 ms |
| `aten::copy_` | 12.50 ms | 12.71 ms | +0.21 ms |
| `aten::mul` | 14.37 ms | 12.65 ms | -1.71 ms |
| `aten::add` | 13.01 ms | 11.19 ms | -1.82 ms |

The largest realized saving is the shorter attention sequence. The largest
remaining cost is GEMM/projection work, followed by sequence packing. Focusing
only on a faster router or a fused gather cannot reach the end-to-end target.

### `addmm` by shape

Shape grouping separates the FFN matrices from width-preserving projections:

| Family | Dense | Sparse cached route | Saving |
| --- | ---: | ---: | ---: |
| FFN up/down projections | 26.26 ms | 17.42 ms | 8.84 ms |
| 5120-to-5120 projection family | 30.78 ms | 27.72 ms | 3.06 ms |

For the 30 sparse middle layers, FFN input length falls from 1785 tokens to 905
tokens (880 selected current-video tokens plus 25 action/state registers). The
ten prefix/suffix layers remain at 1785 tokens.

The width-preserving family still contains 181 dense-length calls in the sparse
trace. Code inspection explains this: current-token sparsity is applied after
self-attention, so self-attention Q/K/V and its output path remain dense across
all current video queries. Sixty middle-layer calls move to length 905, but the
remaining dense projections limit the GEMM saving.

### Packing and gather counts

Sparse cached-route execution has 679 `aten::cat` calls. The dominant batched
cat-copy kernel runs 240 times and consumes about 20.3 ms. Although tensors are
shorter than dense, each layer still repeatedly appends current K/V and packs
query, key, and value tensors for FlashAttention.

Sparse cached-route execution also has exactly 290 `aten::gather` calls:

- 80 historical K/V gathers: two per transformer layer;
- 210 current-block gathers: `x` plus six modulation tensors in each of the 30
  sparse middle layers.

Together these gathers cost about 3.0 ms CUDA time plus about 2.2 ms CPU launch
time. Reducing launch count is worthwhile, but packing/cat is the larger target.

## Router cold versus cached

The cold trace adds six `topk` calls (0.051 ms), two sorts (0.054 ms), one
`amax` (0.006 ms), and the action-key score operations. Summed top `aten::`
self-CUDA time differs by about 0.62 ms between cold and cached traces, which is
an upper bound that also includes normal run-to-run noise.

Spatial `avg_pool2d` remains in the cached trace because radius-one propagation
is applied in selected middle layers; cold routing adds only three more pooling
calls for the three camera regions. Route reuse is therefore already working
and further router-only optimization has limited upside.

## End-to-end server breakdown

The table summarizes the hot same-hardware GPU-5--6 runs, with two warmup and
twelve measured requests. Times are means from the server's internal stage
timers.

| Stage | Dense | Sparse | Saving |
| --- | ---: | ---: | ---: |
| Model total | 1.8125 s | 1.5633 s | 0.2492 s |
| Text encoder | 0.0300 s | 0.0300 s | 0.0000 s |
| Image encoder | 0.0817 s | 0.0800 s | 0.0017 s |
| VAE | 0.0525 s | 0.0525 s | 0.0000 s |
| KV cache creation | 0.1650 s | 0.1642 s | 0.0008 s |
| Diffusion, 8 real DiT steps | 1.4625 s | 1.2217 s | 0.2408 s |
| Scheduler | 0.0200 s | 0.0117 s | 0.0083 s |

Diffusion accounts for 96.6% of the measured model-stage saving. KV creation is
unchanged because the video-only cache-fill path intentionally stays dense.
Transport/transform/untransform overhead is small relative to diffusion and is
not the primary optimization target.

## Optimization priorities

1. **Sparse self-attention queries in middle layers.** Use the existing current
   route for selected video queries plus all registers before FlashAttention,
   while keeping full current K/V for cache correctness. Unselected current
   tokens retain the incoming residual. This directly attacks remaining FA2
   query work and the dense self-attention output projection.
2. **Cache compressed historical K/V per layer across denoise steps.** The route
   is stable for all eight real DiT evaluations and historical KV is immutable;
   gathering it again on every denoise step is redundant.
3. **Replace repeated cat chains with fixed-shape packing.** Preallocate packed
   K/V buffers and fill selected history, dynamic current K/V, and action/state
   registers without creating multiple temporary tensors. Evaluate a fused
   packing kernel only after the buffer layout is fixed.
4. **Collapse current-block metadata gathers.** Avoid seven independent gathers
   per sparse middle layer by carrying modulation in a gather-friendly combined
   representation and by caching register/compute indices.
5. **Then evaluate compile/CUDA graph.** Once shapes and buffers are static,
   graph capture may reduce the many elementwise/copy launches. It should not be
   attempted before removing dynamic allocation and packing churn.
6. **Preserve quality fallback.** Any more aggressive query or layer sparsity
   must keep action/state dense, retain exact full-budget behavior, and use a
   confidence-triggered dense fallback rather than task-specific rules.

## Gate decision

The profiler phase passes: traces are reproducible on both H200s, the real
checkpoint exactness control still passes, and the dominant costs are
identified quantitatively. The next implementation phase should start with
sparse self-attention queries and denoise-persistent historical KV packing,
then rerun operator and checkpoint DiT gates before any end-to-end claim.
