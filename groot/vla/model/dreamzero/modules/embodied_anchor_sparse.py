"""Embodiment-aware spatial anchor routing for causal DreamZero attention.

The Wan VAE and DiT patch embedding preserve a deterministic spatial grid.  For
the released DreamZero-DROID geometry, a 352x640 composite image becomes a
44x80 VAE latent and then a 22x40 DiT grid (880 tokens per latent frame).  This
module uses that correspondence to select control-relevant historical video
tokens without constructing a dense attention matrix.

The initial router is deliberately simple and executable:

* action queries score historical video keys through a small head/dimension
  probe;
* scores are spatially smoothed on the VAE/DiT grid;
* a fixed budget is allocated across embodiment views;
* recent frames stay dense while older frames retain only anchor tokens.

All attention heads share the resulting fixed-length key index.  This is an
important systems constraint: it permits one gathered FlashAttention call per
layer instead of one kernel per head/profile.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ViewRegion:
    """A half-open rectangular region in the DiT token grid."""

    name: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def area(self) -> int:
        return (self.row_end - self.row_start) * (self.col_end - self.col_start)

    def validate(self, grid_height: int, grid_width: int) -> None:
        if not (0 <= self.row_start < self.row_end <= grid_height):
            raise ValueError(f"Invalid row range for {self.name}: {self.row_start}:{self.row_end}")
        if not (0 <= self.col_start < self.col_end <= grid_width):
            raise ValueError(f"Invalid col range for {self.name}: {self.col_start}:{self.col_end}")

    def flat_indices(self, grid_width: int, *, device: torch.device) -> torch.Tensor:
        rows = torch.arange(self.row_start, self.row_end, device=device)
        cols = torch.arange(self.col_start, self.col_end, device=device)
        return (rows[:, None] * grid_width + cols[None, :]).reshape(-1)


def droid_composite_view_regions(
    grid_height: int = 22,
    grid_width: int = 40,
) -> tuple[ViewRegion, ...]:
    """Return the three view regions used by DreamZero-DROID.

    DreamZero constructs a 2x2 composite before VAE encoding.  The wrist image
    spans the complete top row; the two exterior cameras occupy the bottom-left
    and bottom-right quadrants.  The released geometry has an even 22-row token
    grid, so each row of views occupies exactly 11 token rows.
    """

    if grid_height % 2 != 0 or grid_width % 2 != 0:
        raise ValueError("DROID composite routing requires an even token grid")
    half_h = grid_height // 2
    half_w = grid_width // 2
    return (
        ViewRegion("wrist", 0, half_h, 0, grid_width),
        ViewRegion("exterior_left", half_h, grid_height, 0, half_w),
        ViewRegion("exterior_right", half_h, grid_height, half_w, grid_width),
    )


@dataclass(frozen=True)
class AnchorSparseConfig:
    """Configuration for fixed-shape embodiment-aware anchor routing."""

    frame_seqlen: int = 880
    grid_height: int = 22
    grid_width: int = 40
    keep_ratio: float = 0.25
    recent_dense_frames: int = 2
    probe_dim: int = 16
    num_router_heads: int = 4
    smooth_radius: int = 1
    views: tuple[ViewRegion, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_seqlen != self.grid_height * self.grid_width:
            raise ValueError(
                "frame_seqlen must equal grid_height * grid_width, got "
                f"{self.frame_seqlen} vs {self.grid_height}x{self.grid_width}"
            )
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must lie in (0, 1]")
        if self.recent_dense_frames < 0:
            raise ValueError("recent_dense_frames must be non-negative")
        if self.probe_dim <= 0:
            raise ValueError("probe_dim must be positive")
        if self.num_router_heads <= 0:
            raise ValueError("num_router_heads must be positive")
        if self.smooth_radius < 0:
            raise ValueError("smooth_radius must be non-negative")
        for view in self.resolved_views:
            view.validate(self.grid_height, self.grid_width)
        _validate_partition(self.resolved_views, self.grid_height, self.grid_width)

    @property
    def resolved_views(self) -> tuple[ViewRegion, ...]:
        if self.views:
            return self.views
        return (ViewRegion("full", 0, self.grid_height, 0, self.grid_width),)

    @property
    def anchor_tokens_per_frame(self) -> int:
        return max(1, min(self.frame_seqlen, round(self.frame_seqlen * self.keep_ratio)))

    def selected_video_tokens(self, num_frames: int) -> int:
        """Return the fixed routed-video length for ``num_frames`` frames."""

        if num_frames < 0:
            raise ValueError("num_frames must be non-negative")
        num_dense = min(self.recent_dense_frames, num_frames)
        num_sparse = num_frames - num_dense
        return (
            num_sparse * self.anchor_tokens_per_frame
            + num_dense * self.frame_seqlen
        )


@dataclass(frozen=True)
class AnchorRoute:
    """A fixed-shape video-key route for one attention invocation."""

    video_indices: torch.Tensor
    scores: torch.Tensor
    num_video_frames: int
    num_dense_frames: int
    anchor_tokens_per_sparse_frame: int

    @property
    def selected_video_tokens(self) -> int:
        return int(self.video_indices.shape[1])

    def detached(self) -> "AnchorRoute":
        """Return a diagnostic-safe route without autograd references."""

        return AnchorRoute(
            video_indices=self.video_indices.detach(),
            scores=self.scores.detach(),
            num_video_frames=self.num_video_frames,
            num_dense_frames=self.num_dense_frames,
            anchor_tokens_per_sparse_frame=self.anchor_tokens_per_sparse_frame,
        )


def _validate_partition(
    views: Sequence[ViewRegion],
    grid_height: int,
    grid_width: int,
) -> None:
    coverage = torch.zeros((grid_height, grid_width), dtype=torch.int16)
    for view in views:
        coverage[view.row_start : view.row_end, view.col_start : view.col_end] += 1
    if not torch.all(coverage == 1):
        missing = int((coverage == 0).sum())
        overlap = int((coverage > 1).sum())
        raise ValueError(f"View regions must partition the grid exactly: missing={missing}, overlap={overlap}")


def token_indices_to_pixel_boxes(
    token_indices: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Map flat DiT token indices to exact composite-image pixel boxes.

    Returns ``[..., 4]`` boxes in ``(y0, x0, y1, x1)`` half-open format.
    """

    if image_height % grid_height != 0 or image_width % grid_width != 0:
        raise ValueError("Image dimensions must be divisible by the token grid")
    if torch.any(token_indices < 0) or torch.any(token_indices >= grid_height * grid_width):
        raise ValueError("token_indices contain an out-of-grid value")
    patch_h = image_height // grid_height
    patch_w = image_width // grid_width
    rows = torch.div(token_indices, grid_width, rounding_mode="floor")
    cols = token_indices.remainder(grid_width)
    return torch.stack(
        (rows * patch_h, cols * patch_w, (rows + 1) * patch_h, (cols + 1) * patch_w),
        dim=-1,
    )


def action_conditioned_anchor_scores(
    action_query: torch.Tensor,
    video_key: torch.Tensor,
    *,
    probe_dim: int,
    num_router_heads: int,
) -> torch.Tensor:
    """Compute a cheap action-conditioned score for every video key.

    Args:
        action_query: RoPE-applied action queries ``[B, A, H, D]``.
        video_key: RoPE-applied video keys ``[B, K, H, D]``.

    The router uses only a small prefix of heads and head dimensions.  It takes
    the maximum similarity across action tokens and router heads, which retains
    patches strongly requested by any action coordinate without materialising
    the full ``A x K x all_heads x head_dim`` attention tensor.
    """

    if action_query.ndim != 4 or video_key.ndim != 4:
        raise ValueError("action_query and video_key must both have shape [B, L, H, D]")
    if action_query.shape[0] != video_key.shape[0]:
        raise ValueError("action_query and video_key batch sizes differ")
    if action_query.shape[2:] != video_key.shape[2:]:
        raise ValueError("action_query and video_key head shapes differ")

    heads = min(num_router_heads, action_query.shape[2])
    dims = min(probe_dim, action_query.shape[3])
    q = F.normalize(action_query[:, :, :heads, :dims].float(), dim=-1)
    k = F.normalize(video_key[:, :, :heads, :dims].float(), dim=-1)
    similarity = torch.einsum("bahd,bkhd->bahk", q, k)
    return similarity.amax(dim=(1, 2))


def smooth_spatial_scores(
    scores: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    radius: int,
    views: Sequence[ViewRegion] = (),
) -> torch.Tensor:
    """Apply local spatial smoothing independently per frame and camera view."""

    if scores.ndim != 3:
        raise ValueError("scores must have shape [B, F, frame_seqlen]")
    if scores.shape[-1] != grid_height * grid_width:
        raise ValueError("scores do not match the configured spatial grid")
    if radius == 0:
        return scores
    kernel = 2 * radius + 1
    grid = scores.reshape(*scores.shape[:2], grid_height, grid_width)
    if not views:
        views = (ViewRegion("full", 0, grid_height, 0, grid_width),)
    smoothed = torch.empty_like(grid)
    for view in views:
        region = grid[
            ..., view.row_start : view.row_end, view.col_start : view.col_end
        ]
        region_shape = region.shape
        pooled = F.avg_pool2d(
            region.reshape(-1, 1, region_shape[-2], region_shape[-1]),
            kernel_size=kernel,
            stride=1,
            padding=radius,
            count_include_pad=False,
        )
        smoothed[
            ..., view.row_start : view.row_end, view.col_start : view.col_end
        ] = pooled.reshape(region_shape)
    return smoothed.reshape_as(scores)


def _allocate_view_budgets(total_budget: int, capacities: Sequence[int]) -> list[int]:
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if not capacities or any(capacity <= 0 for capacity in capacities):
        raise ValueError("capacities must be a non-empty sequence of positive integers")
    if total_budget > sum(capacities):
        raise ValueError("total_budget exceeds total view capacity")

    total_capacity = sum(capacities)
    raw = [total_budget * capacity / total_capacity for capacity in capacities]
    budgets = [min(capacity, int(value)) for capacity, value in zip(capacities, raw)]

    # When the global budget permits it, every physical view receives at least
    # one token.  This prevents a dominant wrist/exterior view from suppressing
    # all evidence from another camera.
    if total_budget >= len(capacities):
        for index, budget in enumerate(budgets):
            if budget == 0:
                budgets[index] = 1

    remaining = total_budget - sum(budgets)
    remainders = sorted(
        range(len(capacities)),
        key=lambda index: (raw[index] - int(raw[index]), capacities[index]),
        reverse=True,
    )
    while remaining > 0:
        made_progress = False
        for index in remainders:
            if budgets[index] < capacities[index]:
                budgets[index] += 1
                remaining -= 1
                made_progress = True
                if remaining == 0:
                    break
        if not made_progress:
            raise RuntimeError("Could not allocate the requested view budget")

    while remaining < 0:
        made_progress = False
        for index in reversed(remainders):
            floor = 1 if total_budget >= len(capacities) else 0
            if budgets[index] > floor:
                budgets[index] -= 1
                remaining += 1
                made_progress = True
                if remaining == 0:
                    break
        if not made_progress:
            raise RuntimeError("Could not reduce view budgets to the requested total")
    return budgets


def select_view_balanced_anchor_indices(
    scores: torch.Tensor,
    config: AnchorSparseConfig,
) -> torch.Tensor:
    """Select a fixed number of anchor tokens per frame and per view.

    Args:
        scores: Smoothed scores ``[B, F, frame_seqlen]``.

    Returns:
        Sorted frame-local token indices ``[B, F, K]``.
    """

    if scores.ndim != 3 or scores.shape[-1] != config.frame_seqlen:
        raise ValueError("scores must have shape [B, F, frame_seqlen]")
    views = config.resolved_views
    budgets = _allocate_view_budgets(
        config.anchor_tokens_per_frame,
        [view.area for view in views],
    )
    selected = []
    for view, budget in zip(views, budgets):
        region_indices = view.flat_indices(config.grid_width, device=scores.device)
        region_scores = scores.index_select(-1, region_indices)
        local = region_scores.topk(k=budget, dim=-1, sorted=False).indices
        selected.append(region_indices[local])
    return torch.cat(selected, dim=-1).sort(dim=-1).values


def build_current_video_query_route(
    frame_scores: torch.Tensor,
    config: AnchorSparseConfig,
    *,
    keep_ratio: float,
) -> torch.Tensor:
    """Build frame-local anchor indices for current video-query computation.

    Unlike historical KV routing, every current frame uses the same sparse
    budget.  Returned indices address the flattened current-video sequence and
    therefore start at zero, independent of the historical cache length.
    """

    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must lie in (0, 1]")
    if frame_scores.ndim != 3 or frame_scores.shape[-1] != config.frame_seqlen:
        raise ValueError("frame_scores must have shape [B, F, frame_seqlen]")
    batch, num_frames, _ = frame_scores.shape
    if num_frames == 0:
        return torch.empty((batch, 0), dtype=torch.long, device=frame_scores.device)

    query_config = replace(
        config,
        keep_ratio=keep_ratio,
        recent_dense_frames=0,
    )
    local = select_view_balanced_anchor_indices(frame_scores, query_config)
    offsets = (
        torch.arange(num_frames, device=frame_scores.device, dtype=local.dtype)
        * config.frame_seqlen
    )
    return (local + offsets.view(1, num_frames, 1)).reshape(batch, -1)


def build_video_key_route(
    frame_scores: torch.Tensor,
    config: AnchorSparseConfig,
) -> torch.Tensor:
    """Build global video-key indices with dense recent frames.

    ``frame_scores`` has one row per frame in the current video KV cache.  Older
    frames use the anchor budget; the most recent ``recent_dense_frames`` use
    every spatial token.  The returned length is therefore static for a fixed
    number of cached frames.
    """

    if frame_scores.ndim != 3 or frame_scores.shape[-1] != config.frame_seqlen:
        raise ValueError("frame_scores must have shape [B, F, frame_seqlen]")
    batch, num_frames, _ = frame_scores.shape
    if config.anchor_tokens_per_frame == config.frame_seqlen:
        return torch.arange(
            num_frames * config.frame_seqlen,
            device=frame_scores.device,
            dtype=torch.long,
        ).expand(batch, -1)
    num_dense = min(config.recent_dense_frames, num_frames)
    num_sparse = num_frames - num_dense
    chunks = []

    if num_sparse > 0:
        local = select_view_balanced_anchor_indices(frame_scores[:, :num_sparse], config)
        offsets = (
            torch.arange(num_sparse, device=frame_scores.device, dtype=local.dtype)
            * config.frame_seqlen
        )
        chunks.append((local + offsets.view(1, num_sparse, 1)).reshape(batch, -1))

    if num_dense > 0:
        dense_start = num_sparse * config.frame_seqlen
        dense = torch.arange(
            dense_start,
            num_frames * config.frame_seqlen,
            device=frame_scores.device,
            dtype=torch.long,
        ).expand(batch, -1)
        chunks.append(dense)

    if not chunks:
        return torch.empty((batch, 0), dtype=torch.long, device=frame_scores.device)
    return torch.cat(chunks, dim=1)


def route_action_conditioned_video_keys(
    action_query: torch.Tensor,
    video_key: torch.Tensor,
    config: AnchorSparseConfig,
) -> AnchorRoute:
    """Create an embodiment-aware sparse route for a video KV sequence."""

    if video_key.shape[1] % config.frame_seqlen != 0:
        raise ValueError(
            f"Video key length {video_key.shape[1]} is not divisible by frame_seqlen "
            f"{config.frame_seqlen}"
        )
    num_frames = video_key.shape[1] // config.frame_seqlen
    scores = action_conditioned_anchor_scores(
        action_query,
        video_key,
        probe_dim=config.probe_dim,
        num_router_heads=config.num_router_heads,
    ).reshape(video_key.shape[0], num_frames, config.frame_seqlen)
    scores = smooth_spatial_scores(
        scores,
        grid_height=config.grid_height,
        grid_width=config.grid_width,
        radius=config.smooth_radius,
        views=config.resolved_views,
    )
    indices = build_video_key_route(scores, config)
    return AnchorRoute(
        video_indices=indices,
        scores=scores,
        num_video_frames=num_frames,
        num_dense_frames=min(config.recent_dense_frames, num_frames),
        anchor_tokens_per_sparse_frame=config.anchor_tokens_per_frame,
    )


def gather_sequence_by_index(
    sequence: torch.Tensor,
    indices: torch.Tensor,
    *,
    validate_indices: bool = True,
) -> torch.Tensor:
    """Batch-aware gather along the sequence dimension.

    ``sequence`` is ``[B, L, ...]`` and ``indices`` is ``[B, K]``.
    """

    if sequence.ndim < 2 or indices.ndim != 2:
        raise ValueError("sequence must be [B, L, ...] and indices must be [B, K]")
    if sequence.shape[0] != indices.shape[0]:
        raise ValueError("sequence and indices batch sizes differ")
    # The hot attention path passes indices produced by this module and skips
    # this check.  Evaluating a CUDA boolean in Python introduces a device-host
    # synchronization that is disproportionately expensive for the gather.
    if validate_indices and (torch.any(indices < 0) or torch.any(indices >= sequence.shape[1])):
        raise ValueError("indices contain an out-of-range sequence position")
    view_shape = (*indices.shape, *((1,) * (sequence.ndim - 2)))
    expand_shape = (*indices.shape, *sequence.shape[2:])
    expanded = indices.view(view_shape).expand(expand_shape)
    return torch.gather(sequence, dim=1, index=expanded)


def scatter_sequence_by_index(
    sequence: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    *,
    validate_indices: bool = True,
) -> torch.Tensor:
    """Return ``sequence`` with batch-specific sequence positions replaced."""

    if sequence.ndim < 2 or indices.ndim != 2:
        raise ValueError("sequence must be [B, L, ...] and indices must be [B, K]")
    if sequence.shape[0] != indices.shape[0]:
        raise ValueError("sequence and indices batch sizes differ")
    expected_shape = (sequence.shape[0], indices.shape[1], *sequence.shape[2:])
    if updates.shape != expected_shape:
        raise ValueError(
            f"updates must have shape {expected_shape}, got {tuple(updates.shape)}"
        )
    if validate_indices and (torch.any(indices < 0) or torch.any(indices >= sequence.shape[1])):
        raise ValueError("indices contain an out-of-range sequence position")
    view_shape = (*indices.shape, *((1,) * (sequence.ndim - 2)))
    expand_shape = (*indices.shape, *sequence.shape[2:])
    expanded = indices.view(view_shape).expand(expand_shape)
    return sequence.scatter(dim=1, index=expanded, src=updates)


def propagate_spatial_anchor_updates(
    anchor_updates: torch.Tensor,
    anchor_indices: torch.Tensor,
    *,
    video_seq_len: int,
    config: AnchorSparseConfig,
    radius: int,
) -> torch.Tensor:
    """Interpolate anchor residuals to nearby patches within each camera view.

    Selected anchors retain their exact update.  Unselected patches receive the
    normalized mean of selected neighbors in a local window.  Camera regions
    are processed independently, so residuals never bleed across the composite
    image boundaries.
    """

    if radius <= 0:
        raise ValueError("radius must be positive")
    if video_seq_len % config.frame_seqlen != 0:
        raise ValueError("video_seq_len must contain complete frames")
    if anchor_updates.ndim != 3 or anchor_indices.ndim != 2:
        raise ValueError("anchor_updates and anchor_indices must be [B, K, C] and [B, K]")
    if anchor_updates.shape[:2] != anchor_indices.shape:
        raise ValueError("anchor_updates and anchor_indices shapes do not align")

    batch, _, channels = anchor_updates.shape
    num_frames = video_seq_len // config.frame_seqlen
    flat_updates = anchor_updates.new_zeros((batch, video_seq_len, channels))
    flat_updates = scatter_sequence_by_index(
        flat_updates,
        anchor_indices,
        anchor_updates,
        validate_indices=False,
    )
    flat_mask = anchor_updates.new_zeros((batch, video_seq_len, 1))
    flat_mask = scatter_sequence_by_index(
        flat_mask,
        anchor_indices,
        anchor_updates.new_ones((*anchor_indices.shape, 1)),
        validate_indices=False,
    )

    grid_updates = flat_updates.reshape(
        batch,
        num_frames,
        config.grid_height,
        config.grid_width,
        channels,
    ).permute(0, 1, 4, 2, 3).reshape(
        batch * num_frames,
        channels,
        config.grid_height,
        config.grid_width,
    )
    grid_mask = flat_mask.reshape(
        batch,
        num_frames,
        config.grid_height,
        config.grid_width,
        1,
    ).permute(0, 1, 4, 2, 3).reshape(
        batch * num_frames,
        1,
        config.grid_height,
        config.grid_width,
    )
    propagated = torch.empty_like(grid_updates)
    kernel = 2 * radius + 1
    for view in config.resolved_views:
        update_region = grid_updates[
            ..., view.row_start : view.row_end, view.col_start : view.col_end
        ]
        mask_region = grid_mask[
            ..., view.row_start : view.row_end, view.col_start : view.col_end
        ]
        numerator = F.avg_pool2d(
            update_region,
            kernel_size=kernel,
            stride=1,
            padding=radius,
            count_include_pad=False,
        )
        denominator = F.avg_pool2d(
            mask_region,
            kernel_size=kernel,
            stride=1,
            padding=radius,
            count_include_pad=False,
        )
        interpolated = numerator / denominator.clamp_min(1e-6)
        propagated[
            ..., view.row_start : view.row_end, view.col_start : view.col_end
        ] = torch.where(mask_region.bool(), update_region, interpolated)

    return propagated.reshape(
        batch,
        num_frames,
        channels,
        config.grid_height,
        config.grid_width,
    ).permute(0, 1, 3, 4, 2).reshape(batch, video_seq_len, channels)


def route_indices_to_spatial_mask(
    video_indices: torch.Tensor,
    *,
    num_video_frames: int,
    frame_seqlen: int,
) -> torch.Tensor:
    """Convert routed global video indices to a boolean per-frame mask.

    This utility is intended for diagnostics and RGB heatmap generation, not
    the latency-critical attention path.
    """

    if video_indices.ndim != 2:
        raise ValueError("video_indices must have shape [B, K]")
    if num_video_frames < 0 or frame_seqlen <= 0:
        raise ValueError("num_video_frames must be non-negative and frame_seqlen positive")
    total_tokens = num_video_frames * frame_seqlen
    if total_tokens == 0:
        if video_indices.shape[1] != 0:
            raise ValueError("A zero-frame route cannot contain indices")
        return torch.zeros(
            (video_indices.shape[0], 0, frame_seqlen),
            dtype=torch.bool,
            device=video_indices.device,
        )
    if torch.any(video_indices < 0) or torch.any(video_indices >= total_tokens):
        raise ValueError("video_indices contain an out-of-range position")
    flat = torch.zeros(
        (video_indices.shape[0], total_tokens),
        dtype=torch.bool,
        device=video_indices.device,
    )
    flat.scatter_(1, video_indices, True)
    return flat.reshape(video_indices.shape[0], num_video_frames, frame_seqlen)
