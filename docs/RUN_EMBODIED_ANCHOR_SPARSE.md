# Running embodied anchor sparse attention

The upstream checkpoint remains dense by default.  The server configures the
nested causal diffusion model after checkpoint loading, so no checkpoint JSON
or weight rewrite is required.

## H200 server

For the initial four-GPU evaluation allocation:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 uv run --extra gpu python -m torch.distributed.run \
  --standalone --nproc_per_node=4 socket_test_optimized_AR.py \
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
