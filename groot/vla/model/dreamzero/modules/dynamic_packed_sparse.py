"""Packed-state primitives for the Dynamic Sparse Middle Stack.

The hot executor is built around two invariants:

* every lower video budget is a strict prefix of every higher budget; and
* action/state registers precede that video prefix, so changing a budget only
  changes one effective sequence length.

Full-budget execution must bypass the packed executor to preserve the released
Dense path byte-for-byte.  The helpers here nevertheless provide exact tensor
pack/recover behavior and original-position RoPE for validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch

from groot.vla.model.dreamzero.modules.embodied_anchor_sparse import (
    AnchorSparseConfig,
    ViewRegion,
    gather_sequence_by_index,
    scatter_sequence_by_index,
)


@lru_cache(maxsize=32)
def _weighted_view_positions(capacities: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic proportional schedule for nested view prefixes."""

    if not capacities or any(capacity <= 0 for capacity in capacities):
        raise ValueError("capacities must be positive")
    used = [0] * len(capacities)
    positions = [[] for _ in capacities]
    for output_position in range(sum(capacities)):
        candidates = [index for index, capacity in enumerate(capacities) if used[index] < capacity]
        selected = min(
            candidates,
            key=lambda index: (
                (used[index] + 1) / capacities[index],
                -capacities[index],
                index,
            ),
        )
        positions[selected].append(output_position)
        used[selected] += 1
    return tuple(tuple(view_positions) for view_positions in positions)


def nested_view_balanced_ranking(
    scores: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    views: Sequence[ViewRegion],
) -> torch.Tensor:
    """Rank every frame token while keeping every prefix view-balanced.

    Args:
        scores: ``[B, F, H*W]`` frame-local routing scores.

    Returns:
        Frame-local token indices ``[B, F, H*W]``.  Tokens inside each view
        are ordered by descending score; a deterministic weighted schedule
        interleaves the views in proportion to their areas.
    """

    if scores.ndim != 3 or scores.shape[-1] != grid_height * grid_width:
        raise ValueError("scores must have shape [B, F, grid_height*grid_width]")
    if not views:
        views = (ViewRegion("full", 0, grid_height, 0, grid_width),)
    coverage = torch.zeros((grid_height, grid_width), dtype=torch.int8)
    for view in views:
        view.validate(grid_height, grid_width)
        coverage[view.row_start : view.row_end, view.col_start : view.col_end] += 1
    capacities = tuple(view.area for view in views)
    if not torch.all(coverage == 1):
        raise ValueError("views must partition the complete token grid")

    ranked = []
    for view in views:
        region_indices = view.flat_indices(grid_width, device=scores.device)
        local_order = torch.argsort(
            scores.index_select(-1, region_indices),
            dim=-1,
            descending=True,
            stable=True,
        )
        ranked.append(region_indices[local_order])

    output = torch.empty_like(ranked[0]).new_empty(
        (*scores.shape[:2], scores.shape[-1]), dtype=torch.long
    )
    for view_ranking, view_positions in zip(
        ranked, _weighted_view_positions(capacities), strict=True
    ):
        positions = torch.tensor(view_positions, device=scores.device, dtype=torch.long)
        output.index_copy_(-1, positions, view_ranking)
    return output


@dataclass(frozen=True)
class NestedAnchorProfile:
    """One nested route shared by all fixed budget buckets."""

    mandatory_indices: torch.Tensor
    optional_ranked_indices: torch.Tensor
    frame_seqlen: int

    def __post_init__(self) -> None:
        if self.mandatory_indices.ndim != 2:
            raise ValueError("mandatory_indices must have shape [B, M]")
        if self.optional_ranked_indices.ndim != 3:
            raise ValueError("optional_ranked_indices must have shape [B, F, frame_seqlen]")
        if self.mandatory_indices.shape[0] != self.optional_ranked_indices.shape[0]:
            raise ValueError("mandatory and optional routes have different batch sizes")
        if self.optional_ranked_indices.shape[-1] != self.frame_seqlen:
            raise ValueError("optional ranking does not match frame_seqlen")

    @property
    def batch_size(self) -> int:
        return self.mandatory_indices.shape[0]

    @property
    def optional_frames(self) -> int:
        return self.optional_ranked_indices.shape[1]

    def video_tokens_for_ratio(self, keep_ratio: float) -> int:
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must lie in (0, 1]")
        tokens_per_frame = max(
            1, min(self.frame_seqlen, round(self.frame_seqlen * keep_ratio))
        )
        return self.mandatory_indices.shape[1] + self.optional_frames * tokens_per_frame

    def indices_for_ratio(self, keep_ratio: float) -> torch.Tensor:
        tokens_per_frame = (
            self.video_tokens_for_ratio(keep_ratio) - self.mandatory_indices.shape[1]
        ) // max(1, self.optional_frames)
        if self.optional_frames == 0:
            return self.mandatory_indices
        optional = self.optional_ranked_indices[:, :, :tokens_per_frame]
        # Rank-major flattening makes K tokens/frame a strict global prefix of
        # every K'>K route rather than concatenating complete frames.
        optional = optional.transpose(1, 2).reshape(self.batch_size, -1)
        return torch.cat((self.mandatory_indices, optional), dim=1)


def build_nested_current_profile(
    frame_scores: torch.Tensor,
    config: AnchorSparseConfig,
) -> NestedAnchorProfile:
    """Build one nested profile for every current video frame."""

    local = nested_view_balanced_ranking(
        frame_scores,
        grid_height=config.grid_height,
        grid_width=config.grid_width,
        views=config.resolved_views,
    )
    frame_offsets = (
        torch.arange(local.shape[1], device=local.device, dtype=local.dtype)
        * config.frame_seqlen
    )
    global_ranking = local + frame_offsets.view(1, -1, 1)
    mandatory = torch.empty(
        (local.shape[0], 0), device=local.device, dtype=torch.long
    )
    return NestedAnchorProfile(mandatory, global_ranking, config.frame_seqlen)


def build_nested_history_profile(
    frame_scores: torch.Tensor,
    config: AnchorSparseConfig,
) -> NestedAnchorProfile:
    """Build nested historical anchors with recent frames always Dense."""

    if frame_scores.ndim != 3 or frame_scores.shape[-1] != config.frame_seqlen:
        raise ValueError("frame_scores must have shape [B, F, frame_seqlen]")
    batch, num_frames, _ = frame_scores.shape
    dense_frames = min(config.recent_dense_frames, num_frames)
    sparse_frames = num_frames - dense_frames
    if dense_frames:
        dense_start = sparse_frames * config.frame_seqlen
        mandatory = torch.arange(
            dense_start,
            num_frames * config.frame_seqlen,
            device=frame_scores.device,
            dtype=torch.long,
        ).expand(batch, -1)
    else:
        mandatory = torch.empty((batch, 0), device=frame_scores.device, dtype=torch.long)
    if sparse_frames:
        local = nested_view_balanced_ranking(
            frame_scores[:, :sparse_frames],
            grid_height=config.grid_height,
            grid_width=config.grid_width,
            views=config.resolved_views,
        )
        offsets = (
            torch.arange(sparse_frames, device=local.device, dtype=local.dtype)
            * config.frame_seqlen
        )
        optional = local + offsets.view(1, -1, 1)
    else:
        optional = torch.empty(
            (batch, 0, config.frame_seqlen),
            device=frame_scores.device,
            dtype=torch.long,
        )
    return NestedAnchorProfile(mandatory, optional, config.frame_seqlen)


@dataclass
class PackedMiddleState:
    """Current-token state kept packed between transformer layers."""

    frozen_full_x: torch.Tensor
    packed_x: torch.Tensor
    packed_e: tuple[torch.Tensor, ...]
    original_indices: torch.Tensor
    register_tokens: int
    maximum_video_tokens: int

    def active_length(self, video_tokens: int) -> int:
        if not 0 <= video_tokens <= self.maximum_video_tokens:
            raise ValueError("active video budget exceeds the packed route")
        return self.register_tokens + video_tokens

    def active_x(self, video_tokens: int) -> torch.Tensor:
        return self.packed_x[:, : self.active_length(video_tokens)]

    def active_e(self, video_tokens: int) -> tuple[torch.Tensor, ...]:
        length = self.active_length(video_tokens)
        return tuple(part[:, :length] for part in self.packed_e)

    def update_active(self, updated: torch.Tensor, video_tokens: int) -> None:
        length = self.active_length(video_tokens)
        if updated.shape != self.packed_x[:, :length].shape:
            raise ValueError("updated packed state has the wrong shape")
        self.packed_x[:, :length] = updated

    def recover_full(self) -> torch.Tensor:
        return scatter_sequence_by_index(
            self.frozen_full_x,
            self.original_indices,
            self.packed_x,
            validate_indices=False,
        )


def pack_middle_state(
    x: torch.Tensor,
    e: tuple[torch.Tensor, ...],
    profile: NestedAnchorProfile,
    *,
    maximum_keep_ratio: float,
    action_register_length: int,
) -> PackedMiddleState:
    """Gather current registers and maximum video anchors exactly once."""

    if action_register_length <= 0 or action_register_length >= x.shape[1]:
        raise ValueError("action_register_length must identify a non-empty suffix")
    video_seq_len = x.shape[1] - action_register_length
    video_indices = profile.indices_for_ratio(maximum_keep_ratio)
    if torch.any(video_indices >= video_seq_len):
        raise ValueError("profile contains a non-current-video index")
    register_indices = torch.arange(
        video_seq_len,
        x.shape[1],
        device=x.device,
        dtype=torch.long,
    ).expand(x.shape[0], -1)
    # Registers are mandatory and placed first.  A dynamic video budget is
    # therefore represented by one contiguous effective prefix length.
    original_indices = torch.cat((register_indices, video_indices), dim=1)
    packed_x = gather_sequence_by_index(x, original_indices, validate_indices=False)
    packed_e = tuple(
        gather_sequence_by_index(part, original_indices, validate_indices=False)
        for part in e
    )
    return PackedMiddleState(
        frozen_full_x=x,
        packed_x=packed_x,
        packed_e=packed_e,
        original_indices=original_indices,
        register_tokens=action_register_length,
        maximum_video_tokens=video_indices.shape[1],
    )


def gather_packed_rope_frequencies(
    video_freqs: torch.Tensor,
    action_freqs: torch.Tensor,
    state_freqs: torch.Tensor,
    original_indices: torch.Tensor,
    *,
    video_seq_len: int,
    num_action_tokens: int,
    num_state_tokens: int,
    action_state_index: int,
) -> torch.Tensor:
    """Gather RoPE multipliers by each token's original modality/position."""

    action_start = action_state_index * num_action_tokens
    state_start = action_state_index * num_state_tokens
    register_freqs = torch.cat(
        (
            action_freqs[action_start : action_start + num_action_tokens],
            state_freqs[state_start : state_start + num_state_tokens],
        ),
        dim=0,
    )
    full_freqs = torch.cat((video_freqs, register_freqs), dim=0)
    if full_freqs.shape[0] != video_seq_len + num_action_tokens + num_state_tokens:
        raise ValueError("RoPE frequency lengths do not match the packed token layout")
    return gather_sequence_by_index(
        full_freqs.unsqueeze(0).expand(original_indices.shape[0], *full_freqs.shape),
        original_indices,
        validate_indices=True,
    )


def apply_packed_rope(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Apply polar or real-valued RoPE multipliers to a packed tensor."""

    if x.ndim != 4 or frequencies.ndim != 4:
        raise ValueError("x and frequencies must be [B, L, H, D] and [B, L, 1, D/2|D]")
    if x.shape[:2] != frequencies.shape[:2]:
        raise ValueError("packed tokens and frequencies do not align")
    if torch.is_complex(frequencies):
        complex_x = torch.view_as_complex(
            x.to(torch.float64).reshape(*x.shape[:-1], -1, 2)
        )
        return torch.view_as_real(complex_x * frequencies).flatten(3)
    x0, x1 = x.chunk(2, dim=-1)
    frequency_cos, frequency_sin = frequencies.chunk(2, dim=-1)
    return torch.cat(
        (
            x0 * frequency_cos - x1 * frequency_sin,
            x1 * frequency_cos + x0 * frequency_sin,
        ),
        dim=-1,
    )
