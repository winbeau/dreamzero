"""Collect low-overhead Packed-proxy M1 features on real DROID requests."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("DREAMZERO_DISABLE_TORCH_COMPILE", "true")
os.environ.setdefault("ATTENTION_BACKEND", "FA2")

import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch

from benchmarks.collect_dynamic_attention_oracle_droid import (
    _condition_summary,
    _evaluation_modality_configs,
    _init_single_gpu_mesh,
    _load_completed,
    _request_data_point,
    _request_plan,
    _reset_action_head,
    _warmup_history,
)
from groot.vla.data.dataset.lerobot import LeRobotSingleDataset
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.modules.dynamic_m1_observation import (
    PACKED_M1_OBSERVATION_SCHEMA,
    save_packed_m1_observations,
)
from groot.vla.model.dreamzero.modules.dynamic_m1_packed_observer import (
    PackedM1CausalObserver,
)
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


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
    parser.add_argument("--support-ratio", type=float, default=0.20)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument(
        "--warmup-history-blocks",
        type=int,
        default=3,
        choices=range(4),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must lie in [0, num-shards)")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    if not 0.0 < args.support_ratio <= 1.0:
        parser.error("--support-ratio must lie in (0, 1]")
    return args


def _proxy_gate(observations, *, num_layers: int, num_heads: int) -> dict[str, object]:
    valid = [observation for observation in observations if observation is not None]
    gate = {
        "observation_count": len(observations),
        "valid_observation_count": len(valid),
        "dit_indices": [observation.dit_index for observation in valid],
        "schemas": sorted({observation.schema for observation in valid}),
        "shapes": sorted({observation.shape for observation in valid}),
        "first_change_missing": False,
        "later_change_finite": False,
        "first_turnover_missing": False,
        "later_turnover_finite": False,
    }
    if len(valid) == 8:
        first_change = valid[0].metric("packed_action_output_change_relative_l2_max")
        later_change = np.stack(
            [
                observation.metric("packed_action_output_change_relative_l2_max")
                for observation in valid[1:]
            ]
        )
        first_turnover = valid[0].metric("packed_route_support_turnover_max")
        later_turnover = np.stack(
            [
                observation.metric("packed_route_support_turnover_max")
                for observation in valid[1:]
            ]
        )
        gate.update(
            {
                "first_change_missing": bool(np.isnan(first_change).all()),
                "later_change_finite": bool(np.isfinite(later_change).all()),
                "first_turnover_missing": bool(np.isnan(first_turnover).all()),
                "later_turnover_finite": bool(np.isfinite(later_turnover).all()),
            }
        )
    gate["passed"] = bool(
        len(observations) == 8
        and len(valid) == 8
        and gate["dit_indices"] == list(range(8))
        and gate["schemas"] == [PACKED_M1_OBSERVATION_SCHEMA]
        and gate["shapes"] == [(num_layers, num_heads)]
        and gate["first_change_missing"]
        and gate["later_change_finite"]
        and gate["first_turnover_missing"]
        and gate["later_turnover_finite"]
    )
    return gate


def main() -> None:
    args = parse_args()
    os.environ["NUM_DIT_STEPS"] = "8"
    os.environ["DYNAMIC_CACHE_SCHEDULE"] = "False"
    os.environ["ENABLE_DIT_CACHE"] = "false"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output_dir / "proxy"
    artifact_dir.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(
            "Packed proxy protocol requires 16 scheduler and 8 real DiT steps"
        )
    model = action_head.model
    model.configure_dynamic_attention_oracle(output_dir=None)
    model.configure_anchor_sparse_attention(
        enabled=True,
        keep_ratio=1.0,
        current_keep_ratio=1.0,
        dense_prefix_layers=1,
        dense_suffix_layers=1,
        # Recompute only the lightweight action-conditioned Router so support
        # turnover is observed rather than manufactured as zero by cache reuse.
        reuse_denoise=False,
        packed_middle=False,
        record_diagnostics=True,
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
        metadata = {
            "request_key": request_key,
            "split": request["split"],
            "subset_episode_index": subset_episode_index,
            "source_episode_index": source_episode_index,
            "trajectory_stage": stage["name"],
            "trajectory_fraction": float(stage["fraction"]),
            "trajectory_length": trajectory_length,
            "trajectory_step": base_step,
            "instruction_index": instruction_index,
            "length_bucket": request["length_bucket"],
            "manifest_seed": manifest["seed"],
            "scheduler_steps": 16,
            "real_dit_steps": 8,
            "warmup_history_blocks": args.warmup_history_blocks,
            "physical_gpu": args.physical_gpu,
        }
        _reset_action_head(policy)
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
        metadata.update(_condition_summary(data_point))
        metadata["task"] = task
        observer = PackedM1CausalObserver(
            num_layers=model.num_layers,
            num_heads=model.num_heads,
            support_ratio=args.support_ratio,
        )
        model.configure_dynamic_m1_packed_observer(observer)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            policy.lazy_joint_forward_causal(Batch(obs=data_point))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        observations = model.get_dynamic_m1_packed_observations()
        gate = _proxy_gate(
            observations,
            num_layers=model.num_layers,
            num_heads=model.num_heads,
        )
        if not gate["passed"]:
            raise RuntimeError(f"Packed proxy gate failed for {request_key}: {gate}")
        artifact_path = artifact_dir / f"{request_key}.npz"
        save_packed_m1_observations(
            artifact_path,
            observations,
            request_metadata=metadata,
        )
        result = {
            **metadata,
            "elapsed_seconds": elapsed,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "artifact": str(artifact_path),
            "gate": gate,
            "passed": True,
        }
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        model.configure_dynamic_m1_packed_observer(None)
        processed += 1

    summary = {
        "schema": PACKED_M1_OBSERVATION_SCHEMA,
        "physical_gpu": args.physical_gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "planned_requests": len(plan),
        "processed_requests": processed,
        "completed_requests": len(_load_completed(results_path)),
        "artifact_dir": str(artifact_dir),
        "support_ratio": args.support_ratio,
        "scheduler_steps": 16,
        "real_dit_steps": 8,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
