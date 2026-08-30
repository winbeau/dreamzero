from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


STAGE_PATTERN = re.compile(
    r"Time taken: Total (?P<total>[0-9.]+) seconds, "
    r"Text Encoder (?P<text_encoder>[0-9.]+) seconds, "
    r"Image Encoder (?P<image_encoder>[0-9.]+) seconds, "
    r"VAE (?P<vae>[0-9.]+) seconds, "
    r"KV Cache Creation (?P<kv_cache_creation>[0-9.]+) seconds, "
    r"Diffusion (?P<diffusion>[0-9.]+) seconds, "
    r"DIT Compute Steps (?P<dit_steps>[0-9]+) steps, "
    r"Scheduler (?P<scheduler>[0-9.]+) seconds"
)
INFERENCE_PATTERN = re.compile(
    r"Inference Time: Total (?P<total>[0-9.]+) seconds, "
    r"Transform: (?P<transform>[0-9.]+) seconds, "
    r"Model: (?P<model>[0-9.]+) seconds, "
    r"Untransform: (?P<untransform>[0-9.]+) seconds"
)


def _mean_fields(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.fmean(record[key] for record in records)
        for key in records[0]
    }


def summarize_log(
    text: str,
    *,
    total_requests: int,
    warmup_requests: int,
) -> dict[str, object]:
    if total_requests <= 0:
        raise ValueError("total_requests must be positive")
    if warmup_requests < 0 or warmup_requests >= total_requests:
        raise ValueError("warmup_requests must lie in [0, total_requests)")

    stage_records = []
    for match in STAGE_PATTERN.finditer(text):
        record = {
            key: float(value)
            for key, value in match.groupdict().items()
            if key != "dit_steps"
        }
        record["dit_steps"] = int(match.group("dit_steps"))
        stage_records.append(record)
    inference_records = [
        {key: float(value) for key, value in match.groupdict().items()}
        for match in INFERENCE_PATTERN.finditer(text)
    ]

    if len(stage_records) < total_requests or len(inference_records) < total_requests:
        raise ValueError(
            "log does not contain enough complete requests: "
            f"stage={len(stage_records)}, inference={len(inference_records)}, "
            f"required={total_requests}"
        )
    stage_records = stage_records[-total_requests:][warmup_requests:]
    inference_records = inference_records[-total_requests:][warmup_requests:]
    dit_steps = sorted({int(record.pop("dit_steps")) for record in stage_records})
    return {
        "measured_requests": len(stage_records),
        "dit_steps": dit_steps,
        "stage_mean_seconds": _mean_fields(stage_records),
        "inference_mean_seconds": _mean_fields(inference_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the latest complete DreamZero server run in a log."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--total-requests", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize_log(
        args.log.read_text(),
        total_requests=args.total_requests,
        warmup_requests=args.warmup_requests,
    )
    report = {"label": args.label, "log": str(args.log), **summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
