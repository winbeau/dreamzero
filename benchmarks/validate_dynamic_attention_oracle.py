"""Real-checkpoint smoke gate for dense dynamic-attention Oracle capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from groot.vla.model.dreamzero.base_vla import VLA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--history-frames", type=int, default=7)
    parser.add_argument("--layer-indices", type=int, nargs="+", default=[0, 20, 39])
    parser.add_argument("--max-video-queries", type=int, default=2)
    parser.add_argument("--max-action-queries", type=int, default=2)
    parser.add_argument("--query-chunk-size", type=int, default=1)
    return parser.parse_args()


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
    head_dim = model.dim // model.num_heads

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype)

    return {
        "x": randn(batch, model.out_dim, current_frames, latent_height, latent_width),
        "timestep": torch.full(
            (batch, current_frames), 750, device=device, dtype=torch.int64
        ),
        "context": randn(batch, model.text_len, model.text_dim),
        "seq_len": current_frames * model.frame_seqlen,
        "kv_cache": [
            randn(
                2,
                batch,
                history_frames * model.frame_seqlen,
                model.num_heads,
                head_dim,
            )
            for _ in range(model.num_layers)
        ],
        "crossattn_cache": [],
        "current_start_frame": history_frames,
        "y": randn(
            batch,
            model.in_dim - model.out_dim,
            current_frames,
            latent_height,
            latent_width,
        ),
        "clip_feature": randn(batch, 257, 1280),
        "action": randn(batch, model.num_action_per_block, model.action_dim),
        "timestep_action": torch.full(
            (batch, model.num_action_per_block),
            750,
            device=device,
            dtype=torch.int64,
        ),
        "state": randn(batch, model.num_state_per_block, model.max_state_dim),
        "embodiment_id": torch.zeros(batch, device=device, dtype=torch.long),
        "update_kv_cache": False,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = args.output_dir / "capture"

    load_start = time.perf_counter()
    vla = VLA.from_pretrained(str(args.model_path))
    vla.eval().requires_grad_(False)
    model = vla.action_head.model
    load_cpu_seconds = time.perf_counter() - load_start
    model.to(device=device, dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    del vla
    torch.cuda.empty_cache()

    inputs = make_inputs(model, device=device, history_frames=args.history_frames)
    with torch.inference_mode():
        model.configure_anchor_sparse_attention(enabled=False)
        reference_video, reference_action, reference_cache = model(**inputs)
        reference_video = reference_video.clone()
        reference_action = reference_action.clone()
        assert reference_cache == []

        model.configure_dynamic_attention_oracle(
            output_dir=str(capture_dir),
            rank=0,
            max_video_queries=args.max_video_queries,
            max_action_queries=args.max_action_queries,
            query_chunk_size=args.query_chunk_size,
            support_ratio=0.75,
            layer_indices=tuple(args.layer_indices),
            task_id="checkpoint-smoke",
            trajectory_stage="synthetic",
        )
        model.begin_dynamic_attention_oracle_request(
            current_start_frame=args.history_frames,
            instruction="synthetic deterministic checkpoint smoke",
        )
        model.set_dynamic_attention_oracle_step(
            scheduler_index=0,
            dit_index=0,
            scheduler_steps=16,
            timestep=750,
        )
        observed_video, observed_action, observed_cache = model(**inputs)
        capture_paths = model.flush_dynamic_attention_oracle_request()

    if capture_paths is None:
        raise RuntimeError("Oracle collector produced no checkpoint records")
    jsonl_path, profiles_path = capture_paths
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    profiles = torch.load(profiles_path, map_location="cpu", weights_only=True)
    expected_layers = sorted(set(args.layer_indices))
    actual_layers = sorted(record["layer_index"] for record in records)
    result = {
        "physical_gpu": args.physical_gpu,
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(args.model_path),
        "load_cpu_seconds": load_cpu_seconds,
        "history_frames": args.history_frames,
        "current_frames": 2,
        "scheduler_steps": 16,
        "dit_index": 0,
        "layer_indices": expected_layers,
        "record_layers": actual_layers,
        "record_count": len(records),
        "profile_count": len(profiles),
        "max_video_queries": args.max_video_queries,
        "max_action_queries": args.max_action_queries,
        "video_exact": torch.equal(observed_video, reference_video),
        "action_exact": torch.equal(observed_action, reference_action),
        "cache_exact": observed_cache == reference_cache,
        "all_heads_recorded": all(record["num_heads"] == 40 for record in records),
        "all_keep_ratios_recorded": all(
            record["video"]["keep_ratios"]
            == [1.0, 0.75, 0.5, 0.35, 0.25, 0.2, 0.1]
            and record["action"]["keep_ratios"]
            == [1.0, 0.75, 0.5, 0.35, 0.25, 0.2, 0.1]
            for record in records
        ),
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "capture_jsonl": str(jsonl_path),
        "capture_profiles": str(profiles_path),
    }
    result["passed"] = all(
        [
            result["video_exact"],
            result["action_exact"],
            result["cache_exact"],
            result["all_heads_recorded"],
            result["all_keep_ratios_recorded"],
            actual_layers == expected_layers,
            len(profiles) == 2 * len(expected_layers),
        ]
    )
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
