import torch

from groot.vla.model.dreamzero.modules.dynamic_packed_sparse import (
    apply_packed_rope,
    build_nested_current_profile,
    build_nested_history_profile,
    gather_packed_rope_frequencies,
    nested_view_balanced_ranking,
    pack_middle_state,
)
from groot.vla.model.dreamzero.modules.embodied_anchor_sparse import (
    AnchorSparseConfig,
    ViewRegion,
    gather_sequence_by_index,
)
from groot.vla.model.dreamzero.modules.wan2_1_submodule import rope_action_apply


def small_config(*, recent_dense_frames=0):
    return AnchorSparseConfig(
        frame_seqlen=8,
        grid_height=2,
        grid_width=4,
        keep_ratio=0.25,
        recent_dense_frames=recent_dense_frames,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
        views=(
            ViewRegion("top", 0, 1, 0, 4),
            ViewRegion("bottom_left", 1, 2, 0, 2),
            ViewRegion("bottom_right", 1, 2, 2, 4),
        ),
    )


def test_nested_view_ranking_has_unique_complete_prefixes():
    scores = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)

    ranking = nested_view_balanced_ranking(
        scores,
        grid_height=2,
        grid_width=4,
        views=small_config().views,
    )

    assert ranking.shape == (1, 1, 8)
    assert torch.equal(ranking.sort(dim=-1).values, torch.arange(8).reshape(1, 1, 8))
    # The first three tokens cover all three physical views.
    assert set(ranking[0, 0, :3].tolist()) & set(range(4))
    assert set(ranking[0, 0, :3].tolist()) & {4, 5}
    assert set(ranking[0, 0, :3].tolist()) & {6, 7}


def test_current_budget_routes_are_strict_nested_prefixes():
    config = small_config()
    scores = torch.randn(2, 3, 8)
    profile = build_nested_current_profile(scores, config)

    low = profile.indices_for_ratio(0.25)
    medium = profile.indices_for_ratio(0.50)
    full = profile.indices_for_ratio(1.0)

    assert torch.equal(low, medium[:, : low.shape[1]])
    assert torch.equal(medium, full[:, : medium.shape[1]])
    assert torch.equal(full.sort(dim=1).values, torch.arange(24).expand(2, -1))


def test_history_profile_keeps_recent_frame_mandatory():
    config = small_config(recent_dense_frames=1)
    scores = torch.randn(1, 3, 8)
    profile = build_nested_history_profile(scores, config)

    low = profile.indices_for_ratio(0.25)
    high = profile.indices_for_ratio(0.5)

    assert torch.equal(low, high[:, : low.shape[1]])
    assert torch.equal(low[:, :8], torch.arange(16, 24).reshape(1, 8))


def test_full_budget_pack_and_recover_are_exact():
    config = small_config()
    scores = torch.randn(2, 2, 8)
    profile = build_nested_current_profile(scores, config)
    x = torch.randn(2, 20, 6)
    e0 = torch.randn(2, 20, 6, 6)

    packed = pack_middle_state(
        x,
        e0,
        profile,
        maximum_keep_ratio=1.0,
        action_register_length=4,
    )

    assert packed.register_tokens == 4
    assert packed.maximum_video_tokens == 16
    assert torch.equal(packed.original_indices[:, :4], torch.arange(16, 20).expand(2, -1))
    assert torch.equal(packed.recover_full(), x)
    assert torch.equal(packed.active_x(16), packed.packed_x)
    assert torch.equal(packed.active_e0(16), packed.packed_e0)


def test_active_update_changes_only_registers_and_requested_video_prefix():
    config = small_config()
    profile = build_nested_current_profile(torch.randn(1, 2, 8), config)
    x = torch.zeros(1, 20, 3)
    e0 = torch.zeros(1, 20, 6, 3)
    packed = pack_middle_state(
        x,
        e0,
        profile,
        maximum_keep_ratio=0.5,
        action_register_length=4,
    )
    video_tokens = profile.video_tokens_for_ratio(0.25)
    packed.update_active(torch.ones_like(packed.active_x(video_tokens)), video_tokens)
    recovered = packed.recover_full()

    updated_indices = packed.original_indices[:, : packed.active_length(video_tokens)]
    assert torch.equal(
        gather_sequence_by_index(recovered, updated_indices),
        torch.ones(1, packed.active_length(video_tokens), 3),
    )
    assert int((recovered == 1).all(dim=-1).sum()) == packed.active_length(video_tokens)


def test_packed_rope_matches_dense_original_positions():
    torch.manual_seed(7)
    batch, video_tokens, action_tokens, state_tokens = 2, 8, 3, 1
    heads, head_dim = 2, 4
    x = torch.randn(batch, video_tokens + action_tokens + state_tokens, heads, head_dim)
    video_freqs = torch.polar(
        torch.ones(video_tokens, 1, head_dim // 2, dtype=torch.float64),
        torch.randn(video_tokens, 1, head_dim // 2, dtype=torch.float64),
    )
    action_freqs = torch.polar(
        torch.ones(6, head_dim // 2, dtype=torch.float64),
        torch.randn(6, head_dim // 2, dtype=torch.float64),
    )
    state_freqs = torch.polar(
        torch.ones(2, head_dim // 2, dtype=torch.float64),
        torch.randn(2, head_dim // 2, dtype=torch.float64),
    )
    original_indices = torch.tensor([[8, 9, 10, 11, 0, 3, 7], [8, 9, 10, 11, 1, 4, 6]])
    packed_x = gather_sequence_by_index(x, original_indices)
    packed_freqs = gather_packed_rope_frequencies(
        video_freqs,
        action_freqs,
        state_freqs,
        original_indices,
        video_seq_len=video_tokens,
        num_action_tokens=action_tokens,
        num_state_tokens=state_tokens,
        action_state_index=0,
    )
    packed_output = apply_packed_rope(packed_x, packed_freqs)
    dense_output = rope_action_apply(
        x,
        video_freqs,
        action_freqs,
        state_freqs,
        action_register_length=action_tokens + state_tokens,
        num_action_per_block=action_tokens,
        num_state_per_block=state_tokens,
    )

    assert torch.equal(packed_output, gather_sequence_by_index(dense_output, original_indices))
