# Dynamic M1/M2 performance report

Date: 2026-08-30

## Status

This report freezes the Dense and old per-layer gather/scatter sparse DiT
baseline required before Dynamic Packed M2 work.  The fixed-budget Packed
Middle Stack, dynamic M1/M2, 100-request server runs, GPU-group exchange, and
closed-loop measurements are not complete and no final performance claim is
made here.

## Protocol invariant

DreamZero retains the released 16 scheduler steps.  With `NUM_DIT_STEPS=8`,
the fixed mask executes eight real DiT evaluations at scheduler indices
`0, 1, 2, 6, 10, 13, 14, 15`.  Sparse results are not allowed to change this
mask or reduce the number of real DiT calls.

Checkpoint baseline setup:

- checkpoint: released DreamZero-DROID, 16.48B diffusion parameters;
- BF16 FlashAttention 2, eager mode;
- seven cached frames plus two current frames;
- 40 layers and 40 heads;
- old sparse configuration: historical/current/Q keep 20%/20%/20%, two recent
  dense KV frames, dense prefix/suffix 5/5, radius-one propagation every five
  middle layers, route reuse, and exact no-update denoising;
- two warmups and six measured Dense/Sparse forwards on physical H200 GPUs
  2, 3, 5, and 6 concurrently;
- every rank performs a paired Dense then Sparse measurement on the same GPU.

## Frozen old sparse DiT baseline

| Physical GPU | Dense p50 | Old Sparse p50 | Speedup |
| ---: | ---: | ---: | ---: |
| 2 | 187.90 ms | 135.66 ms | 1.385x |
| 3 | 189.46 ms | 134.15 ms | 1.412x |
| 5 | 189.69 ms | 135.92 ms | 1.396x |
| 6 | 186.51 ms | 121.54 ms | 1.535x |
| Mean | 188.39 ms | 131.82 ms | 1.432x |

The ratio of the four mean latencies is 1.429x.  GPU-pair means are 1.399x on
2--3 and 1.465x on 5--6.  The spread, especially GPU 6, proves that later
paper measurements must use the required GPU-group exchange rather than quote
the best device.

All four ranks pass:

- full-budget video/action/all-layer-cache exactness;
- update-disabled versus update-enabled Dense video/action exactness;
- update-disabled versus update-enabled Sparse video/action exactness.

The old 20/20/20 approximation has synthetic video/action relative L2 of
87.14%/11.23% on every rank and therefore remains a speed ceiling, not a
quality-selected policy.

Raw artifacts:

`dynamic_m1_m2/baseline/20260830_old_sparse_gpu2356/`

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=2,3,5,6 DREAMZERO_DISABLE_TORCH_COMPILE=true \
  uv run --frozen --no-sync torchrun \
  --nproc_per_node=4 --master_port=29641 \
  benchmarks/validate_dreamzero_checkpoint.py \
  --model-path /data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse/checkpoints/DreamZero-DROID \
  --output-dir /data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/dynamic_m1_m2/baseline/20260830_old_sparse_gpu2356 \
  --physical-gpus 2 3 5 6 \
  --keep-ratios 0.20 0.20 0.20 0.20 \
  --current-keep-ratios 0.20 0.20 0.20 0.20 \
  --attention-query-keep-ratios 0.20 0.20 0.20 0.20 \
  --dense-prefix-layers 5 --dense-suffix-layers 5 \
  --propagate-radius 1 --propagate-every 5 \
  --reuse-denoise --current-attention --no-update-kv-cache \
  --warmup 2 --repeats 6
```

## Stage evidence

- baseline code revision: `d2999f64a8143285a26fc09f84d241866770948c`;
- raw baseline completed on GPUs 2/3/5/6;
- the fixed Packed M2 row remains pending and must be added before stage 1 is
  considered complete.
