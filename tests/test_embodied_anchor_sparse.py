import torch

from groot.vla.model.dreamzero.modules.embodied_anchor_sparse import (
    AnchorSparseConfig,
    ViewRegion,
    action_conditioned_anchor_scores,
    build_current_video_query_route,
    build_video_key_route,
    droid_composite_view_regions,
    gather_sequence_by_index,
    propagate_spatial_anchor_updates,
    route_indices_to_spatial_mask,
    scatter_sequence_by_index,
    select_view_balanced_anchor_indices,
    smooth_spatial_scores,
    token_indices_to_pixel_boxes,
)


def test_droid_views_partition_released_grid() -> None:
    views = droid_composite_view_regions()
    covered = torch.cat([view.flat_indices(40, device=torch.device("cpu")) for view in views])
    assert covered.numel() == 880
    assert torch.equal(covered.sort().values, torch.arange(880))
    assert [view.area for view in views] == [440, 220, 220]


def test_token_to_composite_pixel_correspondence() -> None:
    boxes = token_indices_to_pixel_boxes(
        torch.tensor([0, 39, 40, 879]),
        grid_height=22,
        grid_width=40,
        image_height=352,
        image_width=640,
    )
    assert torch.equal(
        boxes,
        torch.tensor(
            [
                [0, 0, 16, 16],
                [0, 624, 16, 640],
                [16, 0, 32, 16],
                [336, 624, 352, 640],
            ]
        ),
    )


def test_action_conditioned_probe_recovers_high_similarity_key() -> None:
    query = torch.zeros(1, 2, 2, 4)
    key = torch.zeros(1, 6, 2, 4)
    query[0, 1, 0, 0] = 1.0
    key[0, 4, 0, 0] = 3.0
    key[0, 2, 0, 1] = 2.0
    scores = action_conditioned_anchor_scores(
        query,
        key,
        probe_dim=2,
        num_router_heads=1,
    )
    assert scores.argmax(dim=-1).item() == 4


def test_view_balanced_selection_has_exact_budget() -> None:
    config = AnchorSparseConfig(
        keep_ratio=0.1,
        views=droid_composite_view_regions(),
        smooth_radius=0,
    )
    scores = torch.arange(880, dtype=torch.float32).reshape(1, 1, 880)
    selected = select_view_balanced_anchor_indices(scores, config)
    assert selected.shape == (1, 1, 88)
    # Every view must retain at least one key.
    selected_set = set(selected.flatten().tolist())
    for view in droid_composite_view_regions():
        view_set = set(view.flat_indices(40, device=torch.device("cpu")).tolist())
        assert selected_set & view_set


def test_spatial_smoothing_does_not_cross_camera_boundaries() -> None:
    views = (
        ViewRegion("left", 0, 2, 0, 2),
        ViewRegion("right", 0, 2, 2, 4),
    )
    scores = torch.zeros(1, 1, 8)
    scores[0, 0, 1] = 9.0
    smoothed = smooth_spatial_scores(
        scores,
        grid_height=2,
        grid_width=4,
        radius=1,
        views=views,
    ).reshape(1, 1, 2, 4)
    assert torch.count_nonzero(smoothed[..., :2]) > 0
    assert torch.count_nonzero(smoothed[..., 2:]) == 0


def test_old_frames_are_sparse_and_recent_frame_is_dense() -> None:
    config = AnchorSparseConfig(
        frame_seqlen=4,
        grid_height=2,
        grid_width=2,
        keep_ratio=0.5,
        recent_dense_frames=1,
        smooth_radius=0,
    )
    scores = torch.tensor([[[1.0, 4.0, 2.0, 3.0], [8.0, 7.0, 6.0, 5.0], [0.0, 0.0, 0.0, 0.0]]])
    route = build_video_key_route(scores, config)
    assert route.shape == (1, 8)
    assert torch.equal(route[0, -4:], torch.tensor([8, 9, 10, 11]))
    assert set(route[0, :2].tolist()) == {1, 3}
    assert set(route[0, 2:4].tolist()) == {4, 5}


def test_full_budget_route_is_exact_dense_identity() -> None:
    config = AnchorSparseConfig(
        frame_seqlen=4,
        grid_height=2,
        grid_width=2,
        keep_ratio=1.0,
        recent_dense_frames=1,
        smooth_radius=0,
    )
    scores = torch.randn(2, 3, 4)
    route = build_video_key_route(scores, config)
    expected = torch.arange(12).expand(2, -1)
    assert config.selected_video_tokens(3) == 12
    assert torch.equal(route, expected)

    sequence = torch.randn(2, 12, 3, 5)
    gathered = gather_sequence_by_index(sequence, route, validate_indices=False)
    assert torch.equal(gathered, sequence)


def test_current_query_route_is_sparse_for_every_current_frame() -> None:
    config = AnchorSparseConfig(
        frame_seqlen=4,
        grid_height=2,
        grid_width=2,
        keep_ratio=0.5,
        recent_dense_frames=1,
        smooth_radius=0,
    )
    scores = torch.tensor(
        [[[1.0, 4.0, 2.0, 3.0], [8.0, 7.0, 6.0, 5.0]]]
    )
    route = build_current_video_query_route(scores, config, keep_ratio=0.5)
    assert route.shape == (1, 4)
    assert set(route[0, :2].tolist()) == {1, 3}
    assert set(route[0, 2:].tolist()) == {4, 5}


def test_batch_gather_matches_independent_index_select() -> None:
    sequence = torch.arange(2 * 6 * 3).reshape(2, 6, 3)
    indices = torch.tensor([[5, 1, 3], [0, 4, 2]])
    gathered = gather_sequence_by_index(sequence, indices)
    expected = torch.stack(
        [sequence[0].index_select(0, indices[0]), sequence[1].index_select(0, indices[1])]
    )
    assert torch.equal(gathered, expected)


def test_batch_scatter_replaces_only_selected_positions() -> None:
    sequence = torch.arange(2 * 6 * 3).reshape(2, 6, 3)
    indices = torch.tensor([[5, 1, 3], [0, 4, 2]])
    updates = torch.full((2, 3, 3), -1)
    scattered = scatter_sequence_by_index(sequence, indices, updates)
    for batch in range(2):
        for position in range(6):
            if position in indices[batch].tolist():
                assert torch.equal(scattered[batch, position], torch.full((3,), -1))
            else:
                assert torch.equal(scattered[batch, position], sequence[batch, position])


def test_anchor_update_propagation_preserves_anchors_and_camera_boundaries() -> None:
    views = (
        ViewRegion("left", 0, 2, 0, 2),
        ViewRegion("right", 0, 2, 2, 4),
    )
    config = AnchorSparseConfig(
        frame_seqlen=8,
        grid_height=2,
        grid_width=4,
        keep_ratio=0.25,
        recent_dense_frames=0,
        smooth_radius=0,
        views=views,
    )
    indices = torch.tensor([[0, 7]])
    updates = torch.tensor([[[2.0], [8.0]]])
    propagated = propagate_spatial_anchor_updates(
        updates,
        indices,
        video_seq_len=8,
        config=config,
        radius=1,
    )
    assert propagated.shape == (1, 8, 1)
    assert propagated[0, 0, 0] == 2.0
    assert propagated[0, 7, 0] == 8.0
    assert torch.equal(propagated[0, [0, 1, 4, 5], 0], torch.full((4,), 2.0))
    assert torch.equal(propagated[0, [2, 3, 6, 7], 0], torch.full((4,), 8.0))


def test_route_indices_convert_to_per_frame_spatial_mask() -> None:
    indices = torch.tensor([[1, 3, 4, 7]])
    mask = route_indices_to_spatial_mask(
        indices,
        num_video_frames=2,
        frame_seqlen=4,
    )
    assert mask.shape == (1, 2, 4)
    assert torch.equal(mask[0, 0], torch.tensor([False, True, False, True]))
    assert torch.equal(mask[0, 1], torch.tensor([True, False, False, True]))
