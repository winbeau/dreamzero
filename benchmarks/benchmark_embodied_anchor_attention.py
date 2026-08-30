"""H200 microbenchmark for cached embodied-anchor sparse attention.

The benchmark uses the released DreamZero-14B AR attention geometry.  It
reports the three quantities needed to make the systems decision explicit:

1. dense attention;
2. gathered sparse attention with a precomputed/cached route;
3. one-time action-conditioned route construction.

Route construction is not charged once per layer: the implementation shares
one route across all transformer layers and, optionally, denoising calls for a
causal control block.  The JSON also reports conservative amortized estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from typing import Callable

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from groot.vla.model.dreamzero.modules.embodied_anchor_sparse import (
    AnchorSparseConfig,
    droid_composite_view_regions,
    gather_sequence_by_index,
    route_action_conditioned_video_keys,
)


def _sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0
    ).transpose(1, 2)


def _measure_cuda_ms(fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    p95_index = min(len(samples) - 1, round(0.95 * (len(samples) - 1)))
    return {
        "p50_ms": median(samples),
        "p95_ms": samples[p95_index],
        "mean_ms": sum(samples) / len(samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("dreamzero-fa2", "torch-sdpa"), default="dreamzero-fa2")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--denoise-calls", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--query-video-tokens", type=int, default=1760)
    parser.add_argument("--action-tokens", type=int, default=24)
    parser.add_argument("--state-tokens", type=int, default=1)
    parser.add_argument("--video-key-tokens", type=int, default=7920)
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25])
    parser.add_argument("--recent-dense-frames", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.video_key_tokens % 880 != 0:
        raise ValueError("video-key-tokens must contain complete 880-token frames")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = (args.batch, args.heads, args.head_dim)
    register_tokens = args.action_tokens + args.state_tokens
    q = torch.randn(
        shape[0],
        args.query_video_tokens + register_tokens,
        shape[1],
        shape[2],
        device=device,
        dtype=dtype,
    )
    video_k = torch.randn(shape[0], args.video_key_tokens, shape[1], shape[2], device=device, dtype=dtype)
    video_v = torch.randn_like(video_k)
    register_k = torch.randn(
        shape[0], register_tokens, shape[1], shape[2], device=device, dtype=dtype
    )
    register_v = torch.randn_like(register_k)
    action_q = q[:, args.query_video_tokens : args.query_video_tokens + args.action_tokens]
    dense_k = torch.cat((video_k, register_k), dim=1)
    dense_v = torch.cat((video_v, register_v), dim=1)

    if args.backend == "dreamzero-fa2":
        from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule

        attention_module = AttentionModule(
            num_heads=args.heads,
            head_dim=args.head_dim,
            backend="FA2",
        )
        attention = attention_module
    else:
        attention = _sdpa_attention

    dense = _measure_cuda_ms(
        lambda: attention(q, dense_k, dense_v),
        args.warmup,
        args.repeats,
    )

    rows = []
    num_frames = args.video_key_tokens // 880
    for recent_dense_frames in args.recent_dense_frames:
        for keep_ratio in args.keep_ratios:
            config = AnchorSparseConfig(
                keep_ratio=keep_ratio,
                recent_dense_frames=recent_dense_frames,
                probe_dim=16,
                num_router_heads=4,
                smooth_radius=1,
                views=droid_composite_view_regions(),
            )
            route = route_action_conditioned_video_keys(action_q, video_k, config)
            indices = route.video_indices
            sparse_k = torch.cat(
                (gather_sequence_by_index(video_k, indices, validate_indices=False), register_k),
                dim=1,
            )
            sparse_v = torch.cat(
                (gather_sequence_by_index(video_v, indices, validate_indices=False), register_v),
                dim=1,
            )

            router = _measure_cuda_ms(
                lambda: route_action_conditioned_video_keys(action_q, video_k, config).video_indices,
                args.warmup,
                args.repeats,
            )
            sparse_attention_only = _measure_cuda_ms(
                lambda: attention(q, sparse_k, sparse_v),
                args.warmup,
                args.repeats,
            )
            gathered_sparse = _measure_cuda_ms(
                lambda: attention(
                    q,
                    torch.cat((gather_sequence_by_index(video_k, indices, validate_indices=False), register_k), dim=1),
                    torch.cat((gather_sequence_by_index(video_v, indices, validate_indices=False), register_v), dim=1),
                ),
                args.warmup,
                args.repeats,
            )

            amortization = args.layers * args.denoise_calls
            amortized_p50 = gathered_sparse["p50_ms"] + router["p50_ms"] / amortization
            rows.append(
                {
                    "keep_ratio": keep_ratio,
                    "recent_dense_frames": recent_dense_frames,
                    "selected_video_tokens": config.selected_video_tokens(num_frames),
                    "selected_video_fraction": config.selected_video_tokens(num_frames) / args.video_key_tokens,
                    "router_once": router,
                    "sparse_attention_only": sparse_attention_only,
                    "gather_plus_sparse_attention": gathered_sparse,
                    "cached_route_attention_speedup_p50": dense["p50_ms"] / gathered_sparse["p50_ms"],
                    "amortized_router_plus_attention_p50_ms": amortized_p50,
                    "amortized_attention_path_speedup_p50": dense["p50_ms"] / amortized_p50,
                }
            )

    result = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": str(dtype),
        "backend": args.backend,
        "shape": {
            "batch": args.batch,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "query_video_tokens": args.query_video_tokens,
            "action_tokens": args.action_tokens,
            "state_tokens": args.state_tokens,
            "query_total_tokens": args.query_video_tokens + register_tokens,
            "kv_register_tokens": register_tokens,
            "video_key_tokens": args.video_key_tokens,
            "layers": args.layers,
            "denoise_calls": args.denoise_calls,
        },
        "dense_attention": dense,
        "sparse": rows,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
