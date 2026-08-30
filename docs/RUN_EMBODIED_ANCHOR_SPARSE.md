# Running embodied anchor sparse attention

The upstream checkpoint remains dense by default.  The server configures the
nested causal diffusion model after checkpoint loading, so no checkpoint JSON
or weight rewrite is required.

## H200 server

DreamZero's inference-parallel action head supports one or two ranks.  Do not
launch a single four-rank process group: the CFG prediction exchange is a
two-peer protocol.  To use four GPUs for a matched experiment, run independent
two-rank dense and sparse servers.  The following eager-mode setup uses GPUs
2-3 for dense and 4-5 for sparse:

```bash
export DREAMZERO_DISABLE_TORCH_COMPILE=true

CUDA_VISIBLE_DEVICES=2,3 .venv/bin/torchrun \
  --standalone --nproc_per_node=2 socket_test_optimized_AR.py \
  --port 6000 \
  --model-path /data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse/checkpoints/DreamZero-DROID \
  --enable-dit-cache \
  --attention-backend FA2

CUDA_VISIBLE_DEVICES=4,5 .venv/bin/torchrun \
  --standalone --nproc_per_node=2 socket_test_optimized_AR.py \
  --port 6001 \
  --model-path /data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse/checkpoints/DreamZero-DROID \
  --enable-dit-cache \
  --attention-backend FA2 \
  --anchor-sparse-enabled \
  --anchor-sparse-keep-ratio 0.20 \
  --anchor-sparse-recent-dense-frames 2 \
  --anchor-sparse-probe-dim 16 \
  --anchor-sparse-num-router-heads 4 \
  --anchor-sparse-smooth-radius 1 \
  --anchor-sparse-current-keep-ratio 0.50 \
  --anchor-sparse-dense-prefix-layers 5 \
  --anchor-sparse-dense-suffix-layers 5 \
  --anchor-sparse-propagate-radius 1 \
  --anchor-sparse-propagate-every 5 \
  --anchor-sparse-reuse-denoise
```

The virtual environment is managed by uv; invoking its `torchrun` directly
avoids a resolver check during repeated server launches.  When network access
is needed, enter the remote interactive shell with `proxyon` first.

`DREAMZERO_DISABLE_TORCH_COMPILE=true` skips the explicit TextEncoder,
ImageEncoder, VAE, DiT helper, and scheduler compile wrappers.  This is useful
for matched eager-mode measurement and avoids a large first-request compile.
Omit it when deliberately measuring the compiled deployment path.

Two recent dense frames preserve the complete current DreamZero generation
block; `1` is an aggressive ablation, not the default.  Use
`--anchor-sparse-keep-ratio 0.25` for the more conservative candidate.
Omit `--anchor-sparse-enabled` for the matched dense baseline.  Set both
`--anchor-sparse-keep-ratio 1.0` and
`--anchor-sparse-current-keep-ratio 1.0` for the full-budget exactness control.
The current `0.20 / 0.50 / 5+5 / radius-1-every-5` systems candidate passed the
repeated real-checkpoint DiT speed gate at about 1.26x.  Its task quality must
still be decided by the matched closed-loop protocol in
`docs/CLOSED_LOOP_EVAL_PROTOCOL.md`.

Current-token sparse compute is active only on action-conditioned causal WAM
passes.  The initial/cache-rewind conditioning pass and the video-only KV-fill
pass remain dense because neither exposes a valid action-conditioned route.

`--anchor-sparse-attention-query-keep-ratio` independently controls the
current-video self-attention Q budget when
`--anchor-sparse-current-attention` is enabled.  If omitted, it defaults to
`--anchor-sparse-current-keep-ratio`, preserving the original shared route.
Action/state Q remains dense.  Real-checkpoint measurements found that setting
Q below the current cross-attention/FFN budget did not improve latency, so the
independent option is primarily an ablation control.

## End-to-end server latency

The deterministic request generator sends three 180x320 RGB views, robot
state, a fixed language instruction, and a persistent session ID:

```bash
.venv/bin/python benchmarks/benchmark_dreamzero_server_e2e.py \
  --host 127.0.0.1 \
  --port 6000 \
  --warmup-requests 2 \
  --measured-requests 12 \
  --seed 20260830 \
  --label dense-eager-2gpu-long \
  --output /path/to/dense.json
```

Repeat against port 6001 with the same seed and counts.  On the H200 node, the
matched eager-mode run on 2026-08-30 produced:

| Mode | Mean | Median | P90 |
| --- | ---: | ---: | ---: |
| Dense | 1.817 s | 1.800 s | 1.955 s |
| Sparse | 1.575 s | 1.595 s | 1.623 s |
| Dense / sparse | 1.154x | 1.128x | 1.205x |

The paired geometric-mean speedup was 1.153x; all 12 paired requests were
faster with sparse attention (1.092x-1.222x).  This includes preprocessing,
video KV updates, eight actual DiT compute steps, scheduling, action
untransform, and WebSocket transport.  It is an open-loop server latency test,
not a substitute for the closed-loop task-success protocol.

## Route diagnostics

Add `--anchor-sparse-record-diagnostics` to save the layer-0 action-conditioned
scores and selected video-token indices for every server request.  Files are
written under the run output directory as
`anchor_diagnostics/route_XXXXXX.npz`.

Render a route on a 352x640 DROID RGB composite with:

```bash
uv run python scripts/analysis/visualize_anchor_route.py \
  /path/to/route_000001.npz \
  --rgb /path/to/composite.png \
  --frame-index 0 \
  --output /path/to/anchor_overlay.png
```

White lines show the exact wrist/exterior camera boundaries.  The middle panel
shows the action-conditioned score heatmap; the right panel shows the tokens
actually executed by sparse attention.

## Episode reset

The server calls `clear_anchor_sparse_route_cache()` whenever its policy state
is reset, preventing anchor positions from leaking across scenes.  The cache is
also automatically cleared when the causal frame position rewinds.
