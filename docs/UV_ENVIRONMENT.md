# Reproducible uv environment

DreamZero is managed with uv rather than an ad-hoc activated environment.

## Locked setup

- uv: `/data/chenjiayu/.local/bin/uv` (0.11.14 at setup time)
- Python: 3.11, pinned by `.python-version`
- lock file: `uv.lock`
- PyTorch index: official cu129 wheel index
- default H200 development environment: project dependencies + `dev` + `gpu`
- optional GB200/TensorRT tools: `deployment` extra

Create or reproduce the H200 environment with:

```bash
uv sync --extra dev --extra gpu --python 3.11
```

Run tests and benchmarks without relying on shell activation:

```bash
uv run --extra dev --extra gpu pytest -q \
  tests/test_embodied_anchor_sparse.py \
  tests/test_embodied_anchor_attention_integration.py \
  tests/test_full_checkpoint_loading.py

CUDA_VISIBLE_DEVICES=0 uv run --extra gpu python \
  benchmarks/benchmark_embodied_anchor_attention.py \
  --backend dreamzero-fa2
```

The `gpu` extra uses the official FlashAttention 2.8.3 wheel compiled for
CPython 3.11, PyTorch 2.8, CUDA 12, and CXX11 ABI enabled.  TensorRT and
ModelOpt are excluded from the default environment because the official
DreamZero instructions reserve that path for GB200 deployment.

DeepSpeed is isolated in the `training` extra.  It is not imported by the
inference path, and installing it beside Transformers 4.51.3 can trigger a
`modeling_utils`/OPT circular import before model loading.  Training users can
add it explicitly with `uv sync --extra gpu --extra training`.
