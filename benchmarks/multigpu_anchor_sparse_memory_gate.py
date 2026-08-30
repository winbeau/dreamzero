"""Four-GPU DreamZero anchor-attention memory and execution gate.

This is a task-relevant reservation while a full checkpoint is being staged:
each rank reserves a configurable BF16 parameter-shard budget, materializes the
released DreamZero attention geometry, and continuously executes cached-route
FlashAttention.  It therefore verifies both memory headroom and the distributed
runtime instead of idling with an unrelated allocation.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from groot.vla.model.dreamzero.modules.embodied_anchor_sparse import (
    AnchorSparseConfig,
    droid_composite_view_regions,
    gather_sequence_by_index,
    route_action_conditioned_video_keys,
)
from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--reserve-gib", type=float, default=48.0)
    parser.add_argument("--keep-ratio", type=float, default=0.20)
    parser.add_argument("--recent-dense-frames", type=int, default=2)
    parser.add_argument("--report-interval", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    dtype = torch.bfloat16

    reserve_elements = int(args.reserve_gib * 1024**3 / torch.tensor([], dtype=dtype).element_size())
    parameter_shard = torch.empty(reserve_elements, dtype=dtype, device=device)

    batch, heads, head_dim = 1, 40, 128
    video_queries, action_tokens, state_tokens, video_keys = 1760, 24, 1, 7920
    register_tokens = action_tokens + state_tokens
    q = torch.randn(batch, video_queries + register_tokens, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, video_keys, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    register_k = torch.randn(batch, register_tokens, heads, head_dim, device=device, dtype=dtype)
    register_v = torch.randn_like(register_k)
    config = AnchorSparseConfig(
        keep_ratio=args.keep_ratio,
        recent_dense_frames=args.recent_dense_frames,
        probe_dim=16,
        num_router_heads=4,
        smooth_radius=1,
        views=droid_composite_view_regions(),
    )
    route = route_action_conditioned_video_keys(
        q[:, video_queries : video_queries + action_tokens], k, config
    )
    routed_k = torch.cat(
        (gather_sequence_by_index(k, route.video_indices, validate_indices=False), register_k),
        dim=1,
    )
    routed_v = torch.cat(
        (gather_sequence_by_index(v, route.video_indices, validate_indices=False), register_v),
        dim=1,
    )
    attention = AttentionModule(num_heads=heads, head_dim=head_dim, backend="FA2")

    for _ in range(20):
        attention(q, routed_k, routed_v)
    torch.cuda.synchronize()
    dist.barrier()

    start = time.monotonic()
    iterations = 0
    while time.monotonic() - start < args.duration_seconds:
        output = attention(q, routed_k, routed_v)
        # Touch the parameter allocation and attention output so neither can be
        # optimized away while keeping the extra work negligible.
        parameter_shard[iterations % parameter_shard.numel()] = output.flatten()[0]
        iterations += 1
        if iterations % args.report_interval == 0:
            torch.cuda.synchronize()
            counter = torch.tensor([iterations], dtype=torch.int64, device=device)
            dist.all_reduce(counter, op=dist.ReduceOp.MIN)
            if rank == 0:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                print(
                    {
                        "min_iterations": int(counter.item()),
                        "elapsed_seconds": time.monotonic() - start,
                        "free_gib_rank0": free_bytes / 1024**3,
                        "total_gib_rank0": total_bytes / 1024**3,
                        "selected_video_tokens": route.selected_video_tokens,
                    },
                    flush=True,
                )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
