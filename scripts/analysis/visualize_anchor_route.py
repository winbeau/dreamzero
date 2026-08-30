"""Render action-conditioned DreamZero anchor scores on the RGB composite."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(values[finite], (1.0, 99.0))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _load_rgb(path: Path | None, image_height: int, image_width: int) -> np.ndarray:
    if path is None:
        return np.full((image_height, image_width, 3), 0.35, dtype=np.float32)
    if path.suffix == ".npy":
        image = np.load(path)
    else:
        image = iio.imread(path)
    if image.shape[:2] != (image_height, image_width):
        raise ValueError(
            f"RGB composite must be {image_height}x{image_width}, got {image.shape[:2]}"
        )
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = image[..., :3].astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def _draw_view_boundaries(axis, image_height: int, image_width: int) -> None:
    half_h = image_height // 2
    half_w = image_width // 2
    axis.axhline(half_h - 0.5, color="white", linewidth=1.2, alpha=0.9)
    axis.plot(
        [half_w - 0.5, half_w - 0.5],
        [half_h - 0.5, image_height - 0.5],
        color="white",
        linewidth=1.2,
        alpha=0.9,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path, help="NPZ emitted by the DreamZero server")
    parser.add_argument("--rgb", type=Path, help="Optional 352x640 RGB composite image")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.route)
    scores = data["scores"]
    video_indices = data["video_indices"]
    num_frames = int(data["num_video_frames"])
    grid_height = int(data["grid_height"])
    grid_width = int(data["grid_width"])
    image_height = int(data["image_height"])
    image_width = int(data["image_width"])
    frame_seqlen = grid_height * grid_width

    frame_index = args.frame_index % num_frames
    if not 0 <= args.batch_index < scores.shape[0]:
        raise ValueError(f"batch-index {args.batch_index} is out of range")

    frame_scores = scores[args.batch_index, frame_index].reshape(grid_height, grid_width)
    selected = np.zeros(num_frames * frame_seqlen, dtype=bool)
    selected[video_indices[args.batch_index]] = True
    frame_selected = selected.reshape(num_frames, grid_height, grid_width)[frame_index]

    patch_h = image_height // grid_height
    patch_w = image_width // grid_width
    score_pixels = np.repeat(
        np.repeat(_normalize(frame_scores), patch_h, axis=0),
        patch_w,
        axis=1,
    )
    selected_pixels = np.repeat(
        np.repeat(frame_selected.astype(np.float32), patch_h, axis=0),
        patch_w,
        axis=1,
    )
    rgb = _load_rgb(args.rgb, image_height, image_width)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("DROID RGB composite")
    axes[1].imshow(rgb)
    axes[1].imshow(score_pixels, cmap="inferno", alpha=0.68, vmin=0.0, vmax=1.0)
    axes[1].set_title("Action-conditioned anchor score")
    axes[2].imshow(rgb)
    axes[2].imshow(
        np.ma.masked_where(selected_pixels == 0, selected_pixels),
        cmap="spring",
        alpha=0.62,
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_title(
        f"Executed anchors: frame {frame_index}, "
        f"{int(frame_selected.sum())}/{frame_seqlen} tokens"
    )
    for axis in axes:
        _draw_view_boundaries(axis, image_height, image_width)
        axis.set_xlim(-0.5, image_width - 0.5)
        axis.set_ylim(image_height - 0.5, -0.5)
        axis.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
