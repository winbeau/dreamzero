from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_candidates(
    *,
    dit_indices: tuple[int, ...],
    layer_indices: tuple[int, ...],
    num_heads: int,
    group_size: int,
    query_scopes: tuple[str, ...] = ("all",),
    scale: float = 0.0,
) -> list[dict[str, Any]]:
    if not dit_indices or any(index < 0 for index in dit_indices):
        raise ValueError("dit_indices must be non-empty and non-negative")
    if len(set(dit_indices)) != len(dit_indices):
        raise ValueError("dit_indices must be unique")
    if not layer_indices or any(index < 0 for index in layer_indices):
        raise ValueError("layer_indices must be non-empty and non-negative")
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError("layer_indices must be unique")
    if num_heads <= 0 or group_size <= 0 or num_heads % group_size != 0:
        raise ValueError("group_size must divide the positive num_heads exactly")
    if not query_scopes or any(
        scope not in {"all", "video", "register"} for scope in query_scopes
    ):
        raise ValueError("query_scopes must contain all, video, or register")
    if len(set(query_scopes)) != len(query_scopes):
        raise ValueError("query_scopes must be unique")

    candidates = []
    for dit_index in dit_indices:
        for layer_index in layer_indices:
            for query_scope in query_scopes:
                for head_start in range(0, num_heads, group_size):
                    head_stop = head_start + group_size
                    candidates.append(
                        {
                            "label": (
                                f"d{dit_index}_l{layer_index}_"
                                f"h{head_start:02d}_{head_stop - 1:02d}_"
                                f"{query_scope}"
                            ),
                            "dit_index": dit_index,
                            "layer_index": layer_index,
                            "head_indices": list(range(head_start, head_stop)),
                            "scale": scale,
                            "query_scope": query_scope,
                        }
                    )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed-shape downstream Oracle Head-group candidates over "
            "selected real DiT and Transformer layer cells."
        )
    )
    parser.add_argument("--dit-indices", type=int, nargs="+", required=True)
    parser.add_argument("--layer-indices", type=int, nargs="+", required=True)
    parser.add_argument("--num-heads", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--query-scopes",
        nargs="+",
        choices=("all", "video", "register"),
        default=("all",),
    )
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    candidates = build_candidates(
        dit_indices=tuple(args.dit_indices),
        layer_indices=tuple(args.layer_indices),
        num_heads=args.num_heads,
        group_size=args.group_size,
        query_scopes=tuple(args.query_scopes),
        scale=args.scale,
    )
    payload = {
        "schema_version": 1,
        "description": args.description,
        "dit_indices": args.dit_indices,
        "layer_indices": args.layer_indices,
        "num_heads": args.num_heads,
        "group_size": args.group_size,
        "query_scopes": args.query_scopes,
        "scale": args.scale,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
