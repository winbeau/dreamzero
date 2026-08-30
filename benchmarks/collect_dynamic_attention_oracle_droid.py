"""Collect full Dense Attention Oracle evidence on real DreamZero DROID requests."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("DREAMZERO_DISABLE_TORCH_COMPILE", "true")
os.environ.setdefault("ATTENTION_BACKEND", "FA2")

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tianshou.data import Batch

from groot.vla.data.dataset.lerobot import LeRobotSingleDataset
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


KEEP_RATIOS = (1.0, 0.75, 0.50, 0.35, 0.25, 0.20, 0.10)
EXPECTED_SCHEDULER_INDICES = (0, 1, 2, 6, 10, 13, 14, 15)
LANGUAGE_KEYS = (
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("early", "middle", "late"),
        default=("early", "middle", "late"),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-video-queries", type=int, default=32)
    parser.add_argument("--max-action-queries", type=int)
    parser.add_argument("--query-chunk-size", type=int, default=4)
    parser.add_argument("--support-ratio", type=float, default=0.75)
    parser.add_argument("--layer-indices", type=int, nargs="*", default=())
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument(
        "--warmup-history-blocks",
        type=int,
        default=3,
        choices=range(0, 4),
        help=(
            "Real-observation AR blocks used to build historical KV before capture. "
            "Three blocks produce seven historical latent frames."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must lie in [0, num-shards)")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    return args


def _init_single_gpu_mesh():
    dist.init_process_group("nccl")
    if dist.get_world_size() != 1:
        raise ValueError("Oracle collection uses independent one-GPU workers")
    torch.cuda.set_device(0)
    return init_device_mesh(
        device_type="cuda",
        mesh_shape=(1,),
        mesh_dim_names=("ip",),
    )


def _evaluation_modality_configs(policy: GrootSimPolicy):
    configs = copy.deepcopy(policy.modality_configs)
    for name in ("video", "state", "language"):
        if name not in configs:
            continue
        config = configs[name]
        if config.eval_delta_indices is not None:
            config.delta_indices = list(config.eval_delta_indices)
    return configs


def _reset_action_head(policy: GrootSimPolicy) -> None:
    action_head = policy.trained_model.action_head
    action_head.current_start_frame = 0
    action_head.language = None
    action_head.clip_feas = None
    action_head.ys = None
    action_head.kv_cache1 = None
    action_head.kv_cache_neg = None
    action_head.crossattn_cache = None
    action_head.crossattn_cache_neg = None
    clear_route_cache = getattr(action_head.model, "clear_anchor_sparse_route_cache", None)
    if clear_route_cache is not None:
        clear_route_cache()


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, Batch):
        return {key: _to_cpu(value[key]) for key in value.keys()}
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("passed"):
                completed.add(record["request_key"])
    return completed


def _request_plan(manifest: dict, args: argparse.Namespace) -> list[dict]:
    plan = []
    allowed_splits = set(args.splits)
    allowed_stages = set(args.stages)
    for selection in manifest["selections"]:
        if selection["split"] not in allowed_splits:
            continue
        for stage in selection["trajectory_stages"]:
            if stage["name"] not in allowed_stages:
                continue
            plan.append({**selection, "stage": stage})
    plan = [
        request
        for ordinal, request in enumerate(plan)
        if ordinal % args.num_shards == args.shard_index
    ]
    if args.max_requests is not None:
        plan = plan[: args.max_requests]
    return plan


def _capture_gate(jsonl_path: Path, expected_layers: list[int]) -> dict[str, object]:
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    expected_records = 8 * len(expected_layers) * 2
    actual_layers = sorted({int(record["layer_index"]) for record in records})
    actual_dit_indices = sorted({int(record["dit_index"]) for record in records})
    actual_scheduler_indices = sorted(
        {int(record["scheduler_index"]) for record in records}
    )
    actual_branches = sorted({record["cfg_branch"] for record in records})
    video_key_counts = sorted({int(record["num_video_keys"]) for record in records})
    current_start_frames = sorted(
        {int(record["current_start_frame"]) for record in records}
    )
    gate = {
        "record_count": len(records),
        "expected_record_count": expected_records,
        "layers": actual_layers,
        "dit_indices": actual_dit_indices,
        "scheduler_indices": actual_scheduler_indices,
        "cfg_branches": actual_branches,
        "video_key_counts": video_key_counts,
        "current_start_frames": current_start_frames,
        "all_40_heads": all(record["num_heads"] == 40 for record in records),
        "all_16_scheduler_steps": all(
            record["scheduler_steps"] == 16 for record in records
        ),
        "all_keep_ratios": all(
            record["video"]["keep_ratios"] == list(KEEP_RATIOS)
            and record["action"]["keep_ratios"] == list(KEEP_RATIOS)
            for record in records
        ),
        "all_vv_features": all(
            record["schema_version"] == 3
            and len(record["video_vv_output_change_cosine"]) == 40
            and len(record["video_vv_output_change_relative_l2"]) == 40
            and len(record["action_vv_output_change_cosine"]) == 40
            and len(record["action_vv_output_change_relative_l2"]) == 40
            for record in records
        ),
    }
    gate["passed"] = all(
        (
            len(records) == expected_records,
            actual_layers == expected_layers,
            actual_dit_indices == list(range(8)),
            actual_scheduler_indices == list(EXPECTED_SCHEDULER_INDICES),
            actual_branches == ["conditional", "unconditional"],
            gate["all_40_heads"],
            gate["all_16_scheduler_steps"],
            gate["all_keep_ratios"],
            gate["all_vv_features"],
        )
    )
    return gate


def _pad_leading_video_frames(data_point: dict, expected_frames: int) -> None:
    """Restore repeated boundary frames that timestamp decoding deduplicates."""

    for key, value in data_point.items():
        if not key.startswith("video."):
            continue
        if value.shape[0] > expected_frames:
            raise ValueError(
                f"Video decoder returned {value.shape[0]} frames, expected {expected_frames}"
            )
        missing = expected_frames - value.shape[0]
        if missing:
            if value.shape[0] == 0:
                raise ValueError(f"Video decoder returned no frames for {key}")
            leading = np.repeat(value[:1], missing, axis=0)
            data_point[key] = np.concatenate((leading, value), axis=0)


def _condition_summary(data_point: dict) -> dict[str, float]:
    state_values = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for key, value in data_point.items()
        if key.startswith("state.")
    ]
    action_values = [
        np.asarray(value, dtype=np.float64)
        for key, value in data_point.items()
        if key.startswith("action.")
    ]
    state = np.concatenate(state_values) if state_values else np.zeros(1)
    action = (
        np.concatenate([value.reshape(-1) for value in action_values])
        if action_values
        else np.zeros(1)
    )
    action_delta = (
        np.concatenate(
            [
                (value[-1] - value[0]).reshape(-1)
                for value in action_values
                if value.shape[0] > 0
            ]
        )
        if action_values
        else np.zeros(1)
    )
    return {
        "state_l2": float(np.linalg.vector_norm(state)),
        "state_abs_mean": float(np.mean(np.abs(state))),
        "action_l2": float(np.linalg.vector_norm(action)),
        "action_std": float(np.std(action)),
        "action_temporal_delta_l2": float(np.linalg.vector_norm(action_delta)),
    }


def _request_data_point(
    dataset: LeRobotSingleDataset,
    *,
    trajectory_id: int,
    trajectory_length: int,
    base_step: int,
    video_offsets: tuple[int, ...],
    instruction_index: int,
) -> tuple[dict, str]:
    indices = {}
    for key, delta_indices in dataset.delta_indices.items():
        if key.startswith("video."):
            request_indices = np.asarray(video_offsets, dtype=np.int64) + base_step
        elif key.startswith("annotation."):
            request_indices = np.asarray([base_step], dtype=np.int64)
        else:
            request_indices = delta_indices + base_step
        indices[key] = np.clip(request_indices, 0, trajectory_length - 1)
    data_point = dataset.get_step_data(trajectory_id, indices)
    _pad_leading_video_frames(data_point, len(video_offsets))
    selected_language_key = LANGUAGE_KEYS[instruction_index]
    task = data_point[selected_language_key][0]
    for language_key in LANGUAGE_KEYS:
        data_point.pop(language_key, None)
    data_point[LANGUAGE_KEYS[0]] = [task]
    return data_point, task


def _warmup_history(
    policy: GrootSimPolicy,
    dataset: LeRobotSingleDataset,
    *,
    trajectory_id: int,
    trajectory_length: int,
    base_step: int,
    instruction_index: int,
    history_blocks: int,
) -> str:
    if history_blocks == 0:
        data_point, task = _request_data_point(
            dataset,
            trajectory_id=trajectory_id,
            trajectory_length=trajectory_length,
            base_step=base_step,
            video_offsets=(-3, -2, -1, 0),
            instruction_index=instruction_index,
        )
        sample_metadata.update(_condition_summary(data_point))
        return task

    action_head = policy.trained_model.action_head
    original_mask = list(action_head.dit_step_mask)
    action_head.dit_step_mask = [True] + [False] * 15
    task = ""
    try:
        first_step = base_step - 4 * history_blocks
        warmup_requests = [(first_step, (0,))]
        for block_index in range(1, history_blocks):
            block_end = base_step - 4 * (history_blocks - block_index)
            warmup_requests.append((block_end, (-3, -2, -1, 0)))
        for warmup_step, offsets in warmup_requests:
            data_point, task = _request_data_point(
                dataset,
                trajectory_id=trajectory_id,
                trajectory_length=trajectory_length,
                base_step=warmup_step,
                video_offsets=offsets,
                instruction_index=instruction_index,
            )
            with torch.inference_mode():
                policy.lazy_joint_forward_causal(Batch(obs=data_point))
    finally:
        action_head.dit_step_mask = original_mask
    return task


def main() -> None:
    args = parse_args()
    os.environ["NUM_DIT_STEPS"] = "8"
    os.environ["DYNAMIC_CACHE_SCHEDULE"] = "False"
    os.environ["ENABLE_DIT_CACHE"] = "false"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = args.output_dir / "capture"
    predictions_dir = args.output_dir / "predictions"
    capture_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "request_results.jsonl"
    completed = _load_completed(results_path)
    manifest = json.loads(args.manifest.read_text())
    plan = _request_plan(manifest, args)

    mesh = _init_single_gpu_mesh()
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=str(args.model_path),
        device="cuda",
        device_mesh=mesh,
    )
    policy.eval_transform.eval()
    action_head = policy.trained_model.action_head
    if action_head.num_inference_steps != 16 or sum(action_head.dit_step_mask) != 8:
        raise RuntimeError("Oracle protocol requires 16 scheduler steps and 8 real DiT calls")

    model = action_head.model
    model.configure_anchor_sparse_attention(enabled=False)
    expected_layers = (
        sorted(set(args.layer_indices))
        if args.layer_indices
        else list(range(model.num_layers))
    )
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=_evaluation_modality_configs(policy),
        embodiment_tag=EmbodimentTag("oxe_droid"),
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        use_global_metadata=False,
    )

    processed = 0
    for request in plan:
        subset_episode_index = int(request["subset_episode_index"])
        source_episode_index = int(request["source_episode_index"])
        stage = request["stage"]
        request_key = (
            f"{request['split']}_subset{subset_episode_index:03d}_"
            f"source{source_episode_index:06d}_{stage['name']}"
        )
        if request_key in completed:
            print(f"skip completed {request_key}", flush=True)
            continue

        trajectory_length = int(dataset.trajectory_lengths[subset_episode_index])
        base_step = round((trajectory_length - 1) * float(stage["fraction"]))
        instruction_index = {"early": 0, "middle": 1, "late": 2}[stage["name"]]
        sample_metadata = {
            "request_key": request_key,
            "split": request["split"],
            "subset_episode_index": subset_episode_index,
            "source_episode_index": source_episode_index,
            "trajectory_length": trajectory_length,
            "trajectory_step": base_step,
            "trajectory_fraction": float(stage["fraction"]),
            "instruction_index": instruction_index,
            "length_bucket": request["length_bucket"],
            "manifest_seed": manifest["seed"],
            "warmup_history_blocks": args.warmup_history_blocks,
        }
        _reset_action_head(policy)
        model.configure_dynamic_attention_oracle(output_dir=None)
        torch.manual_seed(args.seed + subset_episode_index)
        torch.cuda.manual_seed_all(args.seed + subset_episode_index)
        np.random.seed(args.seed + subset_episode_index)
        task = _warmup_history(
            policy,
            dataset,
            trajectory_id=subset_episode_index,
            trajectory_length=trajectory_length,
            base_step=base_step,
            instruction_index=instruction_index,
            history_blocks=args.warmup_history_blocks,
        )
        data_point, task = _request_data_point(
            dataset,
            trajectory_id=subset_episode_index,
            trajectory_length=trajectory_length,
            base_step=base_step,
            video_offsets=(-3, -2, -1, 0),
            instruction_index=instruction_index,
        )
        model.configure_dynamic_attention_oracle(
            output_dir=str(capture_dir),
            rank=args.physical_gpu,
            keep_ratios=KEEP_RATIOS,
            max_video_queries=args.max_video_queries,
            max_action_queries=args.max_action_queries,
            query_chunk_size=args.query_chunk_size,
            support_ratio=args.support_ratio,
            layer_indices=tuple(expected_layers),
        )
        model.set_dynamic_attention_oracle_request_metadata(
            task_id=task,
            trajectory_stage=stage["name"],
            sample_metadata=sample_metadata,
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            result_batch, video_pred = policy.lazy_joint_forward_causal(
                Batch(obs=data_point)
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        capture_paths = model.get_dynamic_attention_oracle_last_flush_paths()
        if capture_paths is None:
            raise RuntimeError(f"Oracle produced no capture for {request_key}")
        jsonl_path, profiles_path = capture_paths
        gate = _capture_gate(jsonl_path, expected_layers)
        expected_video_keys = (
            1 + 2 * args.warmup_history_blocks + model.num_frame_per_block
        ) * model.frame_seqlen
        gate["expected_video_keys"] = expected_video_keys
        gate["video_key_count_passed"] = gate["video_key_counts"] == [
            expected_video_keys
        ]
        expected_start_frame = 1 + 2 * args.warmup_history_blocks
        gate["expected_current_start_frame"] = expected_start_frame
        gate["current_start_frame_passed"] = gate["current_start_frames"] == [
            expected_start_frame
        ]
        gate["passed"] = bool(
            gate["passed"]
            and gate["video_key_count_passed"]
            and gate["current_start_frame_passed"]
        )
        if not gate["passed"]:
            raise RuntimeError(f"Oracle capture gate failed for {request_key}: {gate}")

        prediction_path = None
        if not args.no_save_predictions:
            prediction_path = predictions_dir / f"{request_key}.pt"
            torch.save(
                {
                    "request_key": request_key,
                    "task": task,
                    "sample_metadata": sample_metadata,
                    "predicted_action": _to_cpu(result_batch.act),
                    "video_pred": video_pred.detach().cpu(),
                    "ground_truth_action": {
                        key: _to_cpu(value)
                        for key, value in data_point.items()
                        if key.startswith("action.")
                    },
                },
                prediction_path,
            )

        result = {
            "request_key": request_key,
            "task": task,
            **sample_metadata,
            "physical_gpu": args.physical_gpu,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "elapsed_seconds": elapsed,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "capture_jsonl": str(jsonl_path),
            "capture_profiles": str(profiles_path),
            "prediction_path": str(prediction_path) if prediction_path else None,
            "capture_gate": gate,
            "passed": True,
        }
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        processed += 1
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        model.configure_dynamic_attention_oracle(output_dir=None)

    summary = {
        "physical_gpu": args.physical_gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "planned_requests": len(plan),
        "processed_requests": processed,
        "completed_requests": len(_load_completed(results_path)),
        "dataset": str(args.dataset_path),
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "layers": expected_layers,
        "max_video_queries": args.max_video_queries,
        "max_action_queries": args.max_action_queries,
        "query_chunk_size": args.query_chunk_size,
        "support_ratio": args.support_ratio,
        "warmup_history_blocks": args.warmup_history_blocks,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
