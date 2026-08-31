"""Microbenchmark full-width versus head-sliced Packed-M2 attention.

The default geometry matches the released DreamZero-14B AR DROID checkpoint:
40 heads, width 5120, 25 Dense action/state registers, a 75% packed current
trunk, and seven historical latent frames. The benchmark intentionally keeps
the immutable historical-KV gather cache warm, as the production executor
reuses it across denoising calls.
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

from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanSelfAttention,
)


def _measure_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    output = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    assert output is not None
    samples.sort()
    p95_index = min(len(samples) - 1, round(0.95 * (len(samples) - 1)))
    return {
        "samples_ms": samples,
        "p50_ms": median(samples),
        "p95_ms": samples[p95_index],
        "mean_ms": sum(samples) / len(samples),
        "output_abs_mean": float(output.float().abs().mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--register-tokens", type=int, default=25)
    parser.add_argument("--current-video-tokens", type=int, default=1320)
    parser.add_argument("--normal-current-video-tokens", type=int, default=616)
    parser.add_argument("--history-tokens", type=int, default=6160)
    parser.add_argument("--normal-history-tokens", type=int, default=2156)
    parser.add_argument("--critical-heads", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if not 0 < args.critical_heads < args.heads:
        raise ValueError("critical-heads must split the attention heads")
    if args.normal_current_video_tokens > args.current_video_tokens:
        raise ValueError("normal current budget exceeds the packed trunk")
    if args.normal_history_tokens > args.history_tokens:
        raise ValueError("normal history budget exceeds the cache")

    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    dim = args.heads * args.head_dim
    sequence = args.register_tokens + args.current_video_tokens
    attention = CausalWanSelfAttention(
        dim=dim,
        num_heads=args.heads,
        frame_seqlen=880,
        num_action_per_block=args.register_tokens - 1,
        num_state_per_block=1,
    ).to(device=device, dtype=dtype).eval()

    x = torch.randn(1, sequence, dim, device=device, dtype=dtype)
    packed_freqs = torch.zeros(
        1,
        sequence,
        1,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    packed_freqs[..., : args.head_dim // 2] = 1
    kv_cache = torch.randn(
        2,
        1,
        args.history_tokens,
        args.heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    normal_history_indices = torch.arange(
        args.normal_history_tokens,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)

    # Use a non-contiguous critical set so index/weight selection cost matches
    # the learned M1 tables rather than an unrealistically contiguous split.
    interleaved = list(range(0, args.heads, 2)) + list(range(1, args.heads, 2))
    critical_heads = tuple(interleaved[: args.critical_heads])
    critical_set = set(critical_heads)
    normal_heads = tuple(head for head in range(args.heads) if head not in critical_set)

    common = {
        "action_register_length": args.register_tokens,
        "kv_cache": kv_cache,
        "history_indices": normal_history_indices,
        "history_token_count": args.history_tokens,
    }

    def single_sparse_history() -> torch.Tensor:
        attention.packed_dense_action_history = False
        return attention.forward_packed(x, packed_freqs, **common)

    def single_dense_action_history() -> torch.Tensor:
        attention.packed_dense_action_history = True
        return attention.forward_packed(x, packed_freqs, **common)

    def grouped_full_width_current() -> torch.Tensor:
        attention.packed_dense_action_history = False
        return attention.forward_packed(
            x,
            packed_freqs,
            **common,
            head_groups=(
                (critical_heads, 1.0),
                (normal_heads, 0.35),
            ),
            history_indices_by_ratio={0.35: normal_history_indices},
        )

    def grouped_head_sliced_current() -> torch.Tensor:
        attention.packed_dense_action_history = False
        return attention.forward_packed(
            x,
            packed_freqs,
            **common,
            head_groups=(
                (critical_heads, 1.0, 0.75),
                (normal_heads, 0.35, 0.35),
            ),
            history_indices_by_ratio={0.35: normal_history_indices},
            current_video_tokens_by_ratio={
                0.75: args.current_video_tokens,
                0.35: args.normal_current_video_tokens,
            },
        )

    methods = {
        "single_sparse_history_current75": single_sparse_history,
        "single_sparse_video_dense_action_history_current75": (
            single_dense_action_history
        ),
        "grouped_history_only_full_width_current75": grouped_full_width_current,
        "grouped_head_sliced_current75_35": grouped_head_sliced_current,
    }
    results = {}
    with torch.inference_mode():
        for name, fn in methods.items():
            attention.clear_anchor_sparse_history_cache()
            fn()
            torch.cuda.synchronize()
            results[name] = _measure_cuda_ms(
                fn,
                warmup=args.warmup,
                repeats=args.repeats,
            )

    full_width = float(
        results["grouped_history_only_full_width_current75"]["p50_ms"]
    )
    sliced = float(results["grouped_head_sliced_current75_35"]["p50_ms"])
    payload = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "attention_backend": attention.attn.backend,
        "geometry": {
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dim": dim,
            "register_tokens": args.register_tokens,
            "current_video_tokens": args.current_video_tokens,
            "normal_current_video_tokens": args.normal_current_video_tokens,
            "history_tokens": args.history_tokens,
            "normal_history_tokens": args.normal_history_tokens,
            "critical_heads": len(critical_heads),
            "normal_heads": len(normal_heads),
        },
        "results": results,
        "head_sliced_speedup_over_full_width_grouped": full_width / sliced,
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device)
        / 1024**3,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
