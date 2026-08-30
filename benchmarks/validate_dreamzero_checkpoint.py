"""Four-GPU real-checkpoint gate for embodied anchor sparse attention.

Each rank loads the released full DreamZero checkpoint, releases the temporary
GPU reservation only after every rank is ready to move the DiT to CUDA, checks
the exact full-budget invariant, and measures a paired dense/sparse DiT forward
at the released DROID geometry.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import time

import torch
import torch.distributed as dist

from groot.vla.model.dreamzero.base_vla import VLA
from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedHeadGroupBudgetTable,
    DynamicPackedBudgetTable,
)


REAL_DIT_SCHEDULER_INDICES = (0, 1, 2, 6, 10, 13, 14, 15)
REAL_DIT_TIMESTEPS = (999, 986, 972, 892, 749, 535, 416, 249)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--release-pids", type=int, nargs="*", default=[])
    parser.add_argument("--history-frames", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[0.20, 0.25])
    parser.add_argument(
        "--current-keep-ratios", type=float, nargs="+", default=[0.20, 0.25]
    )
    parser.add_argument(
        "--attention-query-keep-ratios",
        type=float,
        nargs="+",
        help="Per-rank self-attention Q keep ratios; defaults to current-keep-ratios.",
    )
    parser.add_argument("--dense-prefix-layers", type=int, default=1)
    parser.add_argument("--dense-suffix-layers", type=int, default=1)
    parser.add_argument(
        "--dense-prefix-layer-candidates",
        type=int,
        nargs="+",
        help="Optional per-rank prefix depths aligned with keep-ratios.",
    )
    parser.add_argument(
        "--dense-suffix-layer-candidates",
        type=int,
        nargs="+",
        help="Optional per-rank suffix depths aligned with keep-ratios.",
    )
    parser.add_argument("--propagate-radius", type=int, default=0)
    parser.add_argument("--propagate-every", type=int, default=1)
    parser.add_argument("--reuse-denoise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--current-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--packed-middle",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dynamic-budget-table", type=Path)
    parser.add_argument("--dynamic-head-group-budget-table", type=Path)
    parser.add_argument("--dynamic-budget-dit-index", type=int, default=0)
    parser.add_argument(
        "--dynamic-budget-dit-indices",
        type=int,
        nargs="+",
        help=(
            "Optional per-rank real-DiT indices aligned with keep-ratios. "
            "This permits an early/late comparison from one checkpoint load."
        ),
    )
    parser.add_argument(
        "--update-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return updated per-layer KV caches during timed forwards.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Optional artifact directory for dense and sparse CPU/CUDA traces.",
    )
    parser.add_argument("--profile-row-limit", type=int, default=200)
    parser.add_argument(
        "--profile-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def release_reservation_processes(pids: list[int]) -> None:
    """Terminate only the explicitly supplied task-owned reservation PIDs."""

    for pid in pids:
        if pid <= 1 or not Path(f"/proc/{pid}").exists():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def wait_for_gpu_headroom(device: torch.device, required_free_gib: float = 80.0) -> None:
    deadline = time.monotonic() + 60.0
    required_bytes = int(required_free_gib * 1024**3)
    while time.monotonic() < deadline:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes >= required_bytes:
            return
        time.sleep(0.25)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    raise RuntimeError(
        "GPU reservation did not release enough memory: "
        f"free={free_bytes / 1024**3:.1f} GiB, total={total_bytes / 1024**3:.1f} GiB"
    )


def make_inputs(
    model: torch.nn.Module,
    *,
    device: torch.device,
    history_frames: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(20260830)
    dtype = torch.bfloat16
    batch = 1
    current_frames = 2
    latent_height, latent_width = 44, 80
    action_tokens, action_dim = model.num_action_per_block, model.action_dim
    state_tokens, state_dim = model.num_state_per_block, model.max_state_dim
    head_dim = model.dim // model.num_heads

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype)

    kv_cache = [
        randn(2, batch, history_frames * model.frame_seqlen, model.num_heads, head_dim)
        for _ in range(model.num_layers)
    ]
    return {
        "x": randn(batch, model.out_dim, current_frames, latent_height, latent_width),
        "timestep": torch.full(
            (batch, current_frames), 750, device=device, dtype=torch.int64
        ),
        "context": randn(batch, model.text_len, model.text_dim),
        "seq_len": current_frames * model.frame_seqlen,
        "kv_cache": kv_cache,
        "crossattn_cache": [],
        "current_start_frame": history_frames,
        "y": randn(batch, model.in_dim - model.out_dim, current_frames, latent_height, latent_width),
        "clip_feature": randn(batch, 257, 1280),
        "action": randn(batch, action_tokens, action_dim),
        "timestep_action": torch.full(
            (batch, action_tokens), 750, device=device, dtype=torch.int64
        ),
        "state": randn(batch, state_tokens, state_dim),
        "embodiment_id": torch.zeros(batch, device=device, dtype=torch.long),
    }


def invoke(model: torch.nn.Module, inputs: dict[str, object]):
    return model(**inputs)


def timed_forwards(
    model: torch.nn.Module,
    inputs: dict[str, object],
    *,
    warmup: int,
    repeats: int,
) -> tuple[list[float], tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]:
    last_output = None
    for _ in range(warmup):
        last_output = invoke(model, inputs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last_output = invoke(model, inputs)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    assert last_output is not None
    return samples, last_output


def _event_value(event: object, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def profile_forward(
    model: torch.nn.Module,
    inputs: dict[str, object],
    *,
    output_prefix: Path,
    row_limit: int,
    record_shapes: bool,
):
    if row_limit <= 0:
        raise ValueError("profile-row-limit must be positive")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        output = invoke(model, inputs)
    torch.cuda.synchronize()

    profiler.export_chrome_trace(str(output_prefix.with_suffix(".trace.json")))
    averages = profiler.key_averages(group_by_input_shape=record_shapes)
    table = averages.table(sort_by="self_cuda_time_total", row_limit=row_limit)
    output_prefix.with_suffix(".table.txt").write_text(table + "\n")

    events = []
    for event in averages:
        events.append(
            {
                "key": event.key,
                "count": int(event.count),
                "self_cpu_time_total_us": _event_value(event, "self_cpu_time_total"),
                "cpu_time_total_us": _event_value(event, "cpu_time_total"),
                "self_cuda_time_total_us": _event_value(
                    event,
                    "self_cuda_time_total",
                    "self_device_time_total",
                ),
                "cuda_time_total_us": _event_value(
                    event,
                    "cuda_time_total",
                    "device_time_total",
                ),
                "input_shapes": event.input_shapes if record_shapes else None,
            }
        )
    events.sort(key=lambda event: event["self_cuda_time_total_us"], reverse=True)
    summary = {
        "sort": "self_cuda_time_total_us",
        "record_shapes": record_shapes,
        "events": events[:row_limit],
    }
    output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return output


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((candidate.float() - reference.float()).reshape(-1))
    denominator = torch.linalg.vector_norm(reference.float().reshape(-1)).clamp_min(1e-12)
    return float((numerator / denominator).item())


def cosine_similarity(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_flat = reference.float().reshape(-1)
    candidate_flat = candidate.float().reshape(-1)
    denominator = (
        torch.linalg.vector_norm(reference_flat)
        * torch.linalg.vector_norm(candidate_flat)
    ).clamp_min(1e-12)
    return float(torch.dot(reference_flat, candidate_flat).div(denominator).item())


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if len(args.physical_gpus) != world_size:
        raise ValueError(
            "physical-gpus must name exactly one device per rank: "
            f"got {args.physical_gpus} for world_size={world_size}"
        )
    if not args.keep_ratios:
        raise ValueError("keep-ratios must contain at least one candidate")
    if len(args.current_keep_ratios) != len(args.keep_ratios):
        raise ValueError("current-keep-ratios must align one-to-one with keep-ratios")
    if args.attention_query_keep_ratios is None:
        args.attention_query_keep_ratios = args.current_keep_ratios
    if len(args.attention_query_keep_ratios) != len(args.keep_ratios):
        raise ValueError(
            "attention-query-keep-ratios must align one-to-one with keep-ratios"
        )
    if args.packed_middle and args.update_kv_cache:
        raise ValueError("Packed Middle Stack timing requires --no-update-kv-cache")
    if args.packed_middle and args.propagate_radius > 0:
        raise ValueError("Packed Middle Stack does not support per-layer propagation")
    if args.dynamic_budget_table is not None and not args.packed_middle:
        raise ValueError("Dynamic budget tables require --packed-middle")
    if args.dynamic_head_group_budget_table is not None and not args.packed_middle:
        raise ValueError("Dynamic head-group budget tables require --packed-middle")
    if not 0 <= args.dynamic_budget_dit_index < 8:
        raise ValueError("dynamic-budget-dit-index must lie in [0, 7]")
    if (
        args.dynamic_budget_dit_indices is not None
        and args.dynamic_budget_table is None
        and args.dynamic_head_group_budget_table is None
    ):
        raise ValueError(
            "dynamic-budget-dit-indices requires a dynamic budget table"
        )
    if args.dynamic_budget_dit_indices is not None:
        if len(args.dynamic_budget_dit_indices) != len(args.keep_ratios):
            raise ValueError(
                "dynamic-budget-dit-indices must align one-to-one with keep-ratios"
            )
        if any(not 0 <= index < 8 for index in args.dynamic_budget_dit_indices):
            raise ValueError("dynamic-budget-dit-indices must lie in [0, 7]")
    if args.packed_middle and (
        args.attention_query_keep_ratios != args.current_keep_ratios
    ):
        raise ValueError(
            "Packed Middle Stack requires attention-query and current keep ratios to match"
        )
    for name, candidates in (
        ("dense-prefix-layer-candidates", args.dense_prefix_layer_candidates),
        ("dense-suffix-layer-candidates", args.dense_suffix_layer_candidates),
    ):
        if candidates is not None and len(candidates) != len(args.keep_ratios):
            raise ValueError(f"{name} must align one-to-one with keep-ratios")
    physical_gpu = args.physical_gpus[local_rank]

    load_start = time.perf_counter()
    vla = VLA.from_pretrained(str(args.model_path))
    vla.eval().requires_grad_(False)
    diffusion_model = vla.action_head.model
    load_cpu_seconds = time.perf_counter() - load_start

    dist.barrier()
    if rank == 0:
        release_reservation_processes(args.release_pids)
    dist.barrier()
    wait_for_gpu_headroom(device)

    move_start = time.perf_counter()
    diffusion_model.to(device=device, dtype=torch.bfloat16)
    diffusion_model.eval().requires_grad_(False)
    move_gpu_seconds = time.perf_counter() - move_start
    del vla
    torch.cuda.empty_cache()

    inputs = make_inputs(
        diffusion_model,
        device=device,
        history_frames=args.history_frames,
    )
    timing_inputs = {**inputs, "update_kv_cache": args.update_kv_cache}
    exact_inputs = {**inputs, "update_kv_cache": True}
    candidate_index = rank % len(args.keep_ratios)
    candidate_keep_ratio = args.keep_ratios[candidate_index]
    candidate_current_keep_ratio = args.current_keep_ratios[candidate_index]
    candidate_attention_query_keep_ratio = args.attention_query_keep_ratios[
        candidate_index
    ]
    candidate_dense_prefix_layers = (
        args.dense_prefix_layer_candidates[candidate_index]
        if args.dense_prefix_layer_candidates is not None
        else args.dense_prefix_layers
    )
    candidate_dense_suffix_layers = (
        args.dense_suffix_layer_candidates[candidate_index]
        if args.dense_suffix_layer_candidates is not None
        else args.dense_suffix_layers
    )
    dynamic_budget_table = (
        DynamicPackedBudgetTable.from_json(args.dynamic_budget_table)
        if args.dynamic_budget_table is not None
        else None
    )
    dynamic_head_group_budget_table = (
        DynamicPackedHeadGroupBudgetTable.from_json(
            args.dynamic_head_group_budget_table
        )
        if args.dynamic_head_group_budget_table is not None
        else None
    )
    dynamic_budget_active = (
        dynamic_budget_table is not None
        or dynamic_head_group_budget_table is not None
    )
    candidate_dynamic_budget_dit_index = (
        args.dynamic_budget_dit_indices[candidate_index]
        if args.dynamic_budget_dit_indices is not None
        else args.dynamic_budget_dit_index
    )
    candidate_diffusion_timestep = (
        REAL_DIT_TIMESTEPS[candidate_dynamic_budget_dit_index]
        if dynamic_budget_active
        else 750
    )
    if dynamic_budget_active:
        inputs["timestep"].fill_(candidate_diffusion_timestep)
        inputs["timestep_action"].fill_(candidate_diffusion_timestep)

    with torch.inference_mode():
        diffusion_model.configure_anchor_sparse_attention(enabled=False)
        dense_samples, dense_output = timed_forwards(
            diffusion_model,
            timing_inputs,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        dense_video, dense_action, _ = dense_output
        dense_video = dense_video.clone()
        dense_action = dense_action.clone()
        if args.profile_dir is not None:
            profile_forward(
                diffusion_model,
                timing_inputs,
                output_prefix=args.profile_dir / f"rank{rank}_gpu{physical_gpu}_dense",
                row_limit=args.profile_row_limit,
                record_shapes=args.profile_record_shapes,
            )

        if args.update_kv_cache:
            dense_exact_video, dense_exact_action, dense_exact_caches = dense_output
            no_update_dense_video_exact = None
            no_update_dense_action_exact = None
        else:
            dense_output = None
            torch.cuda.empty_cache()
            dense_exact_video, dense_exact_action, dense_exact_caches = invoke(
                diffusion_model,
                exact_inputs,
            )
            no_update_dense_video_exact = torch.equal(dense_video, dense_exact_video)
            no_update_dense_action_exact = torch.equal(dense_action, dense_exact_action)

        diffusion_model.configure_anchor_sparse_attention(
            enabled=True,
            keep_ratio=1.0,
            recent_dense_frames=2,
            current_keep_ratio=1.0,
            reuse_denoise=False,
            packed_middle=args.packed_middle,
        )
        full_video, full_action, full_caches = invoke(diffusion_model, exact_inputs)
        full_budget_video_exact = torch.equal(full_video, dense_exact_video)
        full_budget_action_exact = torch.equal(full_action, dense_exact_action)
        full_budget_cache_exact = all(
            torch.equal(full_cache, dense_cache)
            for full_cache, dense_cache in zip(full_caches, dense_exact_caches)
        )
        del (
            full_video,
            full_action,
            full_caches,
            dense_exact_video,
            dense_exact_action,
            dense_exact_caches,
            dense_output,
        )
        torch.cuda.empty_cache()

        diffusion_model.configure_anchor_sparse_attention(
            enabled=True,
            keep_ratio=candidate_keep_ratio,
            recent_dense_frames=2,
            current_keep_ratio=candidate_current_keep_ratio,
            attention_query_keep_ratio=candidate_attention_query_keep_ratio,
            dense_prefix_layers=candidate_dense_prefix_layers,
            dense_suffix_layers=candidate_dense_suffix_layers,
            propagate_radius=args.propagate_radius,
            propagate_every=args.propagate_every,
            reuse_denoise=args.reuse_denoise,
            current_attention=args.current_attention,
            packed_middle=args.packed_middle,
            record_diagnostics=True,
        )
        if dynamic_budget_table is not None:
            diffusion_model.configure_dynamic_packed_budget_table(dynamic_budget_table)
        if dynamic_head_group_budget_table is not None:
            diffusion_model.configure_dynamic_packed_head_group_budget_table(
                dynamic_head_group_budget_table
            )
        if dynamic_budget_active:
            diffusion_model.set_dynamic_attention_oracle_step(
                scheduler_index=REAL_DIT_SCHEDULER_INDICES[
                    candidate_dynamic_budget_dit_index
                ],
                dit_index=candidate_dynamic_budget_dit_index,
                scheduler_steps=16,
                timestep=candidate_diffusion_timestep,
            )
        sparse_samples, sparse_output = timed_forwards(
            diffusion_model,
            timing_inputs,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        sparse_video, sparse_action, _ = sparse_output
        if args.update_kv_cache:
            no_update_sparse_video_exact = None
            no_update_sparse_action_exact = None
        else:
            sparse_exact_video, sparse_exact_action, sparse_exact_caches = invoke(
                diffusion_model,
                exact_inputs,
            )
            no_update_sparse_video_exact = torch.equal(
                sparse_video,
                sparse_exact_video,
            )
            no_update_sparse_action_exact = torch.equal(
                sparse_action,
                sparse_exact_action,
            )
            del sparse_exact_video, sparse_exact_action, sparse_exact_caches
            torch.cuda.empty_cache()
        if args.profile_dir is not None:
            diffusion_model.clear_anchor_sparse_route_cache()
            profile_forward(
                diffusion_model,
                timing_inputs,
                output_prefix=(
                    args.profile_dir / f"rank{rank}_gpu{physical_gpu}_sparse_cold_route"
                ),
                row_limit=args.profile_row_limit,
                record_shapes=args.profile_record_shapes,
            )
            profile_forward(
                diffusion_model,
                timing_inputs,
                output_prefix=(
                    args.profile_dir / f"rank{rank}_gpu{physical_gpu}_sparse_cached_route"
                ),
                row_limit=args.profile_row_limit,
                record_shapes=args.profile_record_shapes,
            )
        route = diffusion_model.get_last_anchor_route()

    dense_p50 = statistics.median(dense_samples)
    sparse_p50 = statistics.median(sparse_samples)
    current_frames = 2
    recent_dense_frames = 2
    dense_history_frames = min(
        args.history_frames,
        max(0, recent_dense_frames - current_frames),
    )
    sparse_history_frames = args.history_frames - dense_history_frames
    packed_history_video_tokens = (
        dense_history_frames * 880
        + sparse_history_frames * max(1, round(880 * candidate_keep_ratio))
    )
    packed_current_video_tokens = (
        current_frames * max(1, round(880 * candidate_current_keep_ratio))
    )
    dynamic_middle_layers: list[int] = []
    dynamic_middle_history_ratios: list[float] = []
    dynamic_middle_current_ratios: list[float] = []
    dynamic_middle_history_tokens: list[int] = []
    dynamic_middle_current_tokens: list[int] = []
    if dynamic_budget_table is not None:
        dynamic_middle_layers = list(
            range(
                candidate_dense_prefix_layers,
                dynamic_budget_table.num_layers - candidate_dense_suffix_layers,
            )
        )
        for layer_index in dynamic_middle_layers:
            history_ratio, current_ratio = dynamic_budget_table.ratios(
                candidate_dynamic_budget_dit_index,
                layer_index,
            )
            dynamic_middle_history_ratios.append(history_ratio)
            dynamic_middle_current_ratios.append(current_ratio)
            dynamic_middle_history_tokens.append(
                dense_history_frames * 880
                + sparse_history_frames * max(1, round(880 * history_ratio))
            )
            dynamic_middle_current_tokens.append(
                current_frames * max(1, round(880 * current_ratio))
            )
        if dynamic_middle_layers:
            # The packed state is gathered once at the largest current budget
            # used by the middle stack. Individual layers consume nested active
            # prefixes.
            packed_history_video_tokens = max(dynamic_middle_history_tokens)
            packed_current_video_tokens = max(dynamic_middle_current_tokens)
    dynamic_head_group_history_ratios: list[list[float]] = []
    if dynamic_head_group_budget_table is not None:
        if dynamic_middle_layers:
            head_group_layers = dynamic_middle_layers
        else:
            head_group_layers = list(
                range(
                    candidate_dense_prefix_layers,
                    dynamic_head_group_budget_table.num_layers
                    - candidate_dense_suffix_layers,
                )
            )
        dynamic_head_group_history_ratios = [
            list(
                dynamic_head_group_budget_table.ratios(
                    candidate_dynamic_budget_dit_index,
                    layer_index,
                )
            )
            for layer_index in head_group_layers
        ]
    result = {
        "rank": rank,
        "physical_gpu": physical_gpu,
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(args.model_path),
        "load_cpu_seconds": load_cpu_seconds,
        "move_gpu_seconds": move_gpu_seconds,
        "history_frames": args.history_frames,
        "current_frames": current_frames,
        "keep_ratio": candidate_keep_ratio,
        "current_keep_ratio": candidate_current_keep_ratio,
        "attention_query_keep_ratio": candidate_attention_query_keep_ratio,
        "recent_dense_frames": recent_dense_frames,
        "dense_prefix_layers": candidate_dense_prefix_layers,
        "dense_suffix_layers": candidate_dense_suffix_layers,
        "propagate_radius": args.propagate_radius,
        "propagate_every": args.propagate_every,
        "reuse_denoise": args.reuse_denoise,
        "current_attention": args.current_attention,
        "packed_middle": args.packed_middle,
        "dynamic_budget_table": (
            str(args.dynamic_budget_table)
            if args.dynamic_budget_table is not None
            else None
        ),
        "dynamic_head_group_budget_table": (
            str(args.dynamic_head_group_budget_table)
            if args.dynamic_head_group_budget_table is not None
            else None
        ),
        "dynamic_budget_dit_index": (
            candidate_dynamic_budget_dit_index
            if dynamic_budget_active
            else None
        ),
        "dynamic_budget_diffusion_timestep": (
            candidate_diffusion_timestep
            if dynamic_budget_active
            else None
        ),
        "dynamic_budget_table_name": (
            dynamic_budget_table.name if dynamic_budget_table is not None else None
        ),
        "dynamic_middle_layers": dynamic_middle_layers,
        "dynamic_middle_history_ratios": dynamic_middle_history_ratios,
        "dynamic_middle_current_ratios": dynamic_middle_current_ratios,
        "dynamic_middle_history_ratio_mean": (
            statistics.fmean(dynamic_middle_history_ratios)
            if dynamic_middle_history_ratios
            else None
        ),
        "dynamic_middle_current_ratio_mean": (
            statistics.fmean(dynamic_middle_current_ratios)
            if dynamic_middle_current_ratios
            else None
        ),
        "dynamic_middle_history_ratio_min": (
            min(dynamic_middle_history_ratios)
            if dynamic_middle_history_ratios
            else None
        ),
        "dynamic_middle_history_ratio_max": (
            max(dynamic_middle_history_ratios)
            if dynamic_middle_history_ratios
            else None
        ),
        "dynamic_middle_current_ratio_min": (
            min(dynamic_middle_current_ratios)
            if dynamic_middle_current_ratios
            else None
        ),
        "dynamic_middle_current_ratio_max": (
            max(dynamic_middle_current_ratios)
            if dynamic_middle_current_ratios
            else None
        ),
        "dynamic_middle_history_tokens": dynamic_middle_history_tokens,
        "dynamic_middle_current_tokens": dynamic_middle_current_tokens,
        "dynamic_head_group_table_name": (
            dynamic_head_group_budget_table.name
            if dynamic_head_group_budget_table is not None
            else None
        ),
        "dynamic_head_group_names": (
            list(dynamic_head_group_budget_table.group_names)
            if dynamic_head_group_budget_table is not None
            else []
        ),
        "dynamic_head_groups": (
            [list(group) for group in dynamic_head_group_budget_table.head_groups]
            if dynamic_head_group_budget_table is not None
            else []
        ),
        "dynamic_head_group_history_ratios": dynamic_head_group_history_ratios,
        "update_kv_cache": args.update_kv_cache,
        "dense_samples_ms": dense_samples,
        "sparse_samples_ms": sparse_samples,
        "dense_p50_ms": dense_p50,
        "sparse_p50_ms": sparse_p50,
        "paired_dit_speedup_p50": dense_p50 / sparse_p50,
        "full_budget_video_exact": full_budget_video_exact,
        "full_budget_action_exact": full_budget_action_exact,
        "full_budget_cache_exact": full_budget_cache_exact,
        "no_update_dense_video_exact": no_update_dense_video_exact,
        "no_update_dense_action_exact": no_update_dense_action_exact,
        "no_update_sparse_video_exact": no_update_sparse_video_exact,
        "no_update_sparse_action_exact": no_update_sparse_action_exact,
        "sparse_video_relative_l2": relative_l2(dense_video, sparse_video),
        "sparse_action_relative_l2": relative_l2(dense_action, sparse_action),
        "sparse_video_cosine": cosine_similarity(dense_video, sparse_video),
        "sparse_action_cosine": cosine_similarity(dense_action, sparse_action),
        "selected_video_tokens": route.selected_video_tokens if route is not None else None,
        "selected_current_query_tokens": (
            current_frames
            * max(1, round(880 * candidate_attention_query_keep_ratio))
            if (
                args.packed_middle
                or (
                    args.current_attention
                    and candidate_attention_query_keep_ratio < 1.0
                )
            )
            else 1760
        ),
        "num_video_frames": route.num_video_frames if route is not None else None,
        "packed_history_video_tokens": packed_history_video_tokens,
        "packed_current_video_tokens": packed_current_video_tokens,
        "packed_action_state_tokens": 25,
        "packed_attention_query_tokens": packed_current_video_tokens + 25,
        "packed_attention_key_value_tokens": (
            packed_history_video_tokens + packed_current_video_tokens + 25
        ),
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"rank{rank}_gpu{physical_gpu}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
