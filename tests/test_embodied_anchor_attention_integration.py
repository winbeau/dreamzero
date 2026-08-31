import os
import sys
import types

import torch
import torch.nn as nn

from groot.vla.model.dreamzero.modules.embodied_anchor_sparse import (
    AnchorRoute,
    AnchorSparseConfig,
)
from groot.vla.model.dreamzero.modules.dynamic_sparse_budget import (
    DynamicPackedHeadGroupBudgetTable,
    DynamicPackedBudgetTable,
)


def _load_attention_module():
    """Load the attention file without requiring the full deployment environment."""

    os.environ["ATTENTION_BACKEND"] = "torch"
    try:
        import diffusers  # noqa: F401
    except ModuleNotFoundError:
        configuration_utils = types.ModuleType("diffusers.configuration_utils")
        modeling_utils = types.ModuleType("diffusers.models.modeling_utils")

        class ConfigMixin:
            pass

        class ModelMixin(nn.Module):
            pass

        def register_to_config(function):
            return function

        configuration_utils.ConfigMixin = ConfigMixin
        configuration_utils.register_to_config = register_to_config
        modeling_utils.ModelMixin = ModelMixin
        sys.modules["diffusers"] = types.ModuleType("diffusers")
        sys.modules["diffusers.configuration_utils"] = configuration_utils
        sys.modules["diffusers.models"] = types.ModuleType("diffusers.models")
        sys.modules["diffusers.models.modeling_utils"] = modeling_utils

    import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as module

    # This test exercises KV routing and attention wiring, not RoPE indexing.
    module.causal_rope_action_apply = lambda x, **kwargs: x
    return module


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(1, 7, 8)
    cache = torch.randn(2, 1, 8, 2, 4)
    freqs = torch.ones(32, 2, dtype=torch.complex64)
    return x, cache, freqs


def test_current_token_routing_requires_causal_action_pass() -> None:
    module = _load_attention_module()

    assert not module.action_conditioned_causal_routing_is_eligible(0, 25)
    assert not module.action_conditioned_causal_routing_is_eligible(1, None)
    assert module.action_conditioned_causal_routing_is_eligible(1, 25)


def test_cached_route_is_reused_by_real_kv_attention_path() -> None:
    module = _load_attention_module()
    config = AnchorSparseConfig(
        frame_seqlen=4,
        grid_height=2,
        grid_width=2,
        keep_ratio=0.5,
        recent_dense_frames=1,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
    )
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
        anchor_sparse_config=config,
    )
    x, cache, freqs = _inputs()

    output, updated_cache, route = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )
    cached_output, _, reused_route = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
        anchor_route_indices=route,
    )

    assert output.shape == (1, 7, 8)
    assert updated_cache.shape == (2, 1, 12, 2, 4)
    assert route.shape == (1, 8)
    assert reused_route is route
    assert torch.equal(cached_output, output)


def test_no_update_dense_attention_skips_cache_without_changing_output() -> None:
    module = _load_attention_module()
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    x, cache, freqs = _inputs()

    output, updated_cache, _ = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )
    no_update_output, no_update_cache, _ = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
        update_kv_cache=False,
    )

    assert updated_cache is not None
    assert no_update_cache is None
    assert torch.equal(no_update_output, output)


def test_packed_attention_full_current_budget_matches_dense_path() -> None:
    module = _load_attention_module()
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    x, cache, freqs = _inputs()
    dense_output, _, _ = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
        update_kv_cache=False,
    )
    packed_freqs = torch.ones(
        (1, x.shape[1], 1, attention.head_dim // 2),
        dtype=torch.complex128,
    )
    packed_output = attention.forward_packed(
        x,
        packed_freqs,
        action_register_length=3,
        kv_cache=cache,
        history_indices=torch.arange(cache.shape[2]).reshape(1, -1),
        history_token_count=cache.shape[2],
    )

    assert torch.allclose(packed_output, dense_output, atol=1e-6, rtol=1e-6)


def test_packed_attention_projects_only_effective_tokens():
    module = _load_attention_module()
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    _, cache, _ = _inputs()
    packed_x = torch.randn(1, 5, 8)
    projected_lengths = []
    hooks = [
        projection.register_forward_pre_hook(
            lambda _module, inputs: projected_lengths.append(inputs[0].shape[1])
        )
        for projection in (attention.q, attention.k, attention.v)
    ]
    try:
        output = attention.forward_packed(
            packed_x,
            torch.ones((1, 5, 1, 2), dtype=torch.complex128),
            action_register_length=3,
            kv_cache=cache,
            history_indices=torch.tensor([[1, 3, 5, 7]]),
            history_token_count=cache.shape[2],
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert output.shape == packed_x.shape
    assert projected_lengths == [5, 5, 5]


def test_packed_head_groups_use_distinct_history_lengths_and_dense_registers():
    module = _load_attention_module()
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    _, cache, _ = _inputs()
    packed_x = torch.randn(1, 5, 8)
    history_indices = torch.tensor([[1, 3, 5, 7]])
    attention_shapes = []
    hook = attention.attn.register_forward_pre_hook(
        lambda _module, inputs: attention_shapes.append(
            (inputs[0].shape[1], inputs[1].shape[1], inputs[0].shape[2])
        )
    )
    try:
        output = attention.forward_packed(
            packed_x,
            torch.ones((1, 5, 1, 2), dtype=torch.complex128),
            action_register_length=3,
            kv_cache=cache,
            history_indices=history_indices,
            history_token_count=cache.shape[2],
            head_groups=(((0,), 1.0), ((1,), 0.5)),
            history_indices_by_ratio={0.5: history_indices},
        )
    finally:
        hook.remove()

    assert output.shape == packed_x.shape
    assert attention_shapes == [(5, 13, 1), (5, 9, 1)]
    assert len(attention._anchor_sparse_history_cache) == 1


def test_equal_budget_head_groups_match_single_packed_attention_call():
    module = _load_attention_module()
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    _, cache, _ = _inputs()
    packed_x = torch.randn(1, 5, 8)
    packed_freqs = torch.ones((1, 5, 1, 2), dtype=torch.complex128)
    history_indices = torch.tensor([[1, 3, 5, 7]])

    single = attention.forward_packed(
        packed_x,
        packed_freqs,
        action_register_length=3,
        kv_cache=cache,
        history_indices=history_indices,
        history_token_count=cache.shape[2],
    )
    grouped = attention.forward_packed(
        packed_x,
        packed_freqs,
        action_register_length=3,
        kv_cache=cache,
        history_indices=history_indices,
        history_token_count=cache.shape[2],
        head_groups=(((0,), 0.5), ((1,), 0.5)),
        history_indices_by_ratio={0.5: history_indices},
    )

    assert torch.allclose(grouped, single, atol=1e-6, rtol=1e-6)


def test_packed_head_groups_compress_current_qkv_but_keep_registers_dense():
    module = _load_attention_module()
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    _, cache, _ = _inputs()
    packed_x = torch.randn(1, 5, 8)
    history_indices = torch.tensor([[1, 3, 5, 7]])
    attention_shapes = []
    hook = attention.attn.register_forward_pre_hook(
        lambda _module, inputs: attention_shapes.append(
            (inputs[0].shape[1], inputs[1].shape[1], inputs[0].shape[2])
        )
    )
    try:
        output = attention.forward_packed(
            packed_x,
            torch.ones((1, 5, 1, 2), dtype=torch.complex128),
            action_register_length=3,
            kv_cache=cache,
            history_indices=history_indices,
            history_token_count=cache.shape[2],
            head_groups=(((0,), 1.0, None), ((1,), 0.5, 0.5)),
            history_indices_by_ratio={0.5: history_indices},
            current_video_tokens_by_ratio={0.5: 1},
        )
    finally:
        hook.remove()

    assert output.shape == packed_x.shape
    # The sparse group sees all three leading action/state registers plus one
    # video token as queries and current K/V.
    assert attention_shapes == [(5, 13, 1), (4, 8, 1)]


def test_head_sliced_current_qkv_projects_dense_registers_and_sparse_video():
    module = _load_attention_module()
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    _, cache, _ = _inputs()
    packed_x = torch.randn(1, 5, 8)
    history_indices = torch.tensor([[1, 3, 5, 7]])
    projected_lengths = []
    hooks = [
        projection.register_forward_pre_hook(
            lambda _module, inputs: projected_lengths.append(inputs[0].shape[1])
        )
        for projection in (attention.q, attention.k, attention.v)
    ]
    attention_shapes = []
    attention_hook = attention.attn.register_forward_pre_hook(
        lambda _module, inputs: attention_shapes.append(
            (inputs[0].shape[1], inputs[1].shape[1], inputs[0].shape[2])
        )
    )
    try:
        output = attention.forward_packed(
            packed_x,
            torch.ones((1, 5, 1, 2), dtype=torch.complex128),
            action_register_length=3,
            kv_cache=cache,
            history_indices=history_indices,
            history_token_count=cache.shape[2],
            head_groups=(((0,), 1.0, 1.0), ((1,), 0.5, 0.5)),
            history_indices_by_ratio={0.5: history_indices},
            current_video_tokens_by_ratio={1.0: 2, 0.5: 1},
        )
    finally:
        for hook in hooks:
            hook.remove()
        attention_hook.remove()

    assert output.shape == packed_x.shape
    # Q/K/V modules execute only for the three Dense registers. Video GEMMs
    # use channel-sliced F.linear calls with two and one selected video token.
    assert projected_lengths == [3, 3, 3]
    assert attention_shapes == [(5, 13, 1), (4, 8, 1)]


def test_head_sliced_full_current_groups_match_single_call_without_qk_norm():
    module = _load_attention_module()
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
        qk_norm=False,
    )
    _, cache, _ = _inputs()
    packed_x = torch.randn(1, 5, 8)
    packed_freqs = torch.ones((1, 5, 1, 2), dtype=torch.complex128)
    history_indices = torch.tensor([[1, 3, 5, 7]])

    single = attention.forward_packed(
        packed_x,
        packed_freqs,
        action_register_length=3,
        kv_cache=cache,
        history_indices=history_indices,
        history_token_count=cache.shape[2],
    )
    grouped = attention.forward_packed(
        packed_x,
        packed_freqs,
        action_register_length=3,
        kv_cache=cache,
        history_indices=history_indices,
        history_token_count=cache.shape[2],
        head_groups=(((0,), 0.5, 1.0), ((1,), 0.5, 1.0)),
        history_indices_by_ratio={0.5: history_indices},
        current_video_tokens_by_ratio={1.0: 2},
    )

    assert torch.allclose(grouped, single, atol=1e-6, rtol=1e-6)


def test_packed_attention_does_not_expand_dense_history_window() -> None:
    module = _load_attention_module()
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    cache = torch.arange(2 * 1 * 12 * 2 * 4, dtype=torch.float32).reshape(
        2, 1, 12, 2, 4
    )
    history_indices = torch.tensor([[0, 7]])

    attention.forward_packed(
        torch.randn(1, 5, 8),
        torch.ones((1, 5, 1, 2), dtype=torch.complex128),
        action_register_length=3,
        kv_cache=cache,
        history_indices=history_indices,
        history_token_count=8,
    )

    expected = cache[0, :, -8:][:, [0, 7]]
    assert torch.equal(attention._anchor_sparse_history_k, expected)


def test_complete_packed_block_matches_dense_at_full_budget() -> None:
    module = _load_attention_module()

    class MeanContextCrossAttention(nn.Module):
        def forward(self, x, context):
            return context.mean(dim=1, keepdim=True).expand_as(x)

    dense = module.CausalWanAttentionBlock(
        cross_attn_type="i2v_cross_attn",
        dim=8,
        ffn_dim=16,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    packed = module.CausalWanAttentionBlock(
        cross_attn_type="i2v_cross_attn",
        dim=8,
        ffn_dim=16,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    packed.load_state_dict(dense.state_dict())
    dense.cross_attn = MeanContextCrossAttention()
    packed.cross_attn = MeanContextCrossAttention()
    x, cache, freqs = _inputs()
    e0 = torch.randn(1, x.shape[1], 6, 8)
    context = torch.randn(1, 5, 8)

    dense_output, dense_cache, _ = dense(
        x,
        e0,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        context=context,
        kv_cache=cache,
        current_start_frame=1,
        update_kv_cache=False,
    )
    packed_output = packed.forward_packed(
        x,
        e0,
        torch.ones((1, x.shape[1], 1, 2), dtype=torch.complex128),
        action_register_length=3,
        context=context,
        kv_cache=cache,
        history_indices=torch.arange(cache.shape[2]).reshape(1, -1),
        history_token_count=cache.shape[2],
    )

    assert dense_cache is None
    assert torch.allclose(packed_output, dense_output, atol=1e-6, rtol=1e-6)


def test_oracle_observes_dense_video_action_layout_without_changing_output() -> None:
    module = _load_attention_module()
    dense = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    observed = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    observed.load_state_dict(dense.state_dict())
    x, cache, freqs = _inputs()

    class Collector:
        def __init__(self):
            self.shapes = None

        def observe(self, **kwargs):
            self.shapes = {
                key: tuple(value.shape)
                for key, value in kwargs.items()
                if isinstance(value, torch.Tensor)
            }

    collector = Collector()
    observed.dynamic_oracle_collector = collector
    observed.layer_index = 7
    dense_output, dense_cache, _ = dense(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )
    observed_output, observed_cache, _ = observed(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )

    assert torch.equal(observed_output, dense_output)
    assert torch.equal(observed_cache, dense_cache)
    assert collector.shapes == {
        "video_query": (1, 4, 2, 4),
        "action_query": (1, 2, 2, 4),
        "video_key": (1, 12, 2, 4),
        "video_value": (1, 12, 2, 4),
    }


def test_no_update_sparse_attention_reuses_gathered_history() -> None:
    module = _load_attention_module()
    config = AnchorSparseConfig(
        frame_seqlen=4,
        grid_height=2,
        grid_width=2,
        keep_ratio=0.5,
        recent_dense_frames=1,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
    )
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
        anchor_sparse_config=config,
    )
    x, cache, freqs = _inputs()

    output, _, route = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )
    no_update_output, no_update_cache, reused_route = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
        anchor_route_indices=route,
        update_kv_cache=False,
    )
    selected_history_k = attention._anchor_sparse_history_k
    assert selected_history_k is not None

    cached_output, _, _ = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
        anchor_route_indices=route,
        update_kv_cache=False,
    )

    assert no_update_cache is None
    assert reused_route is route
    assert torch.equal(no_update_output, output)
    assert torch.equal(cached_output, output)
    assert attention._anchor_sparse_history_k is selected_history_k


def test_state_register_is_dense_but_not_used_as_router_query(monkeypatch) -> None:
    module = _load_attention_module()
    config = AnchorSparseConfig(
        frame_seqlen=4,
        grid_height=2,
        grid_width=2,
        keep_ratio=0.5,
        recent_dense_frames=1,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
    )
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
        anchor_sparse_config=config,
    )
    x, cache, freqs = _inputs()
    captured = {}
    original_router = module.route_action_conditioned_video_keys

    def capture_router(*, action_query, video_key, config):
        captured["query_tokens"] = action_query.shape[1]
        return original_router(
            action_query=action_query,
            video_key=video_key,
            config=config,
        )

    monkeypatch.setattr(module, "route_action_conditioned_video_keys", capture_router)
    output, _, _ = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )

    assert output.shape == (1, 7, 8)
    assert captured["query_tokens"] == 2


def test_full_budget_uses_exact_original_dense_path() -> None:
    module = _load_attention_module()
    dense = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    full_budget = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
        anchor_sparse_config=AnchorSparseConfig(
            frame_seqlen=4,
            grid_height=2,
            grid_width=2,
            keep_ratio=1.0,
            recent_dense_frames=1,
            smooth_radius=0,
        ),
    )
    full_budget.load_state_dict(dense.state_dict())
    x, cache, freqs = _inputs()

    dense_output, dense_cache, dense_route = dense(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )
    full_output, full_cache, full_route = full_budget(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )

    assert dense_route is None
    assert full_route is None
    assert torch.equal(full_output, dense_output)
    assert torch.equal(full_cache, dense_cache)


def test_sparse_current_attention_keeps_full_cache_and_register_outputs() -> None:
    module = _load_attention_module()
    config = AnchorSparseConfig(
        frame_seqlen=4,
        grid_height=2,
        grid_width=2,
        keep_ratio=0.5,
        recent_dense_frames=1,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
    )
    attention = module.CausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=4,
        num_action_per_block=2,
        num_state_per_block=1,
        anchor_sparse_config=config,
    )
    x, cache, freqs = _inputs()
    dense_output, dense_cache, route = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
    )
    current_video_indices = torch.tensor([[1, 3]])
    sparse_output, sparse_cache, reused_route = attention(
        x,
        freqs,
        freqs,
        freqs,
        action_register_length=3,
        kv_cache=cache,
        current_start_frame=1,
        anchor_route_indices=route,
        current_video_indices=current_video_indices,
    )

    assert sparse_output.shape == dense_output.shape
    assert torch.equal(sparse_cache, dense_cache)
    assert reused_route is route
    assert torch.equal(sparse_output[:, [0, 2]], torch.zeros_like(sparse_output[:, [0, 2]]))
    assert torch.allclose(sparse_output[:, [1, 3, 4, 5, 6]], dense_output[:, [1, 3, 4, 5, 6]])


def test_post_checkpoint_configuration_updates_every_block() -> None:
    module = _load_attention_module()
    model = module.CausalWanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        frame_seqlen=880,
        text_len=4,
        in_dim=2,
        dim=8,
        ffn_dim=16,
        freq_dim=8,
        text_dim=8,
        out_dim=2,
        num_heads=2,
        num_layers=2,
        action_dim=2,
        max_state_dim=4,
        hidden_size=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )

    model.configure_anchor_sparse_attention(
        enabled=True,
        keep_ratio=0.2,
        current_keep_ratio=0.25,
        attention_query_keep_ratio=0.125,
        dense_prefix_layers=1,
        dense_suffix_layers=0,
        propagate_radius=1,
        propagate_every=1,
        current_attention=True,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
        record_diagnostics=True,
    )

    assert model.anchor_sparse_enabled
    assert model.anchor_sparse_config is not None
    assert model.anchor_sparse_config.anchor_tokens_per_frame == 176
    assert model.anchor_sparse_config.recent_dense_frames == 2
    assert model.anchor_sparse_current_keep_ratio == 0.25
    assert model.anchor_sparse_attention_query_keep_ratio == 0.125
    assert all(
        block.self_attn.anchor_sparse_config is model.anchor_sparse_config
        for block in model.blocks
    )
    assert model.blocks[0].self_attn.record_anchor_diagnostics
    assert not model.blocks[1].self_attn.record_anchor_diagnostics
    assert not model.blocks[0].sparse_current_compute
    assert model.blocks[1].sparse_current_compute
    assert not model.blocks[0].sparse_current_attention
    assert model.blocks[1].sparse_current_attention
    assert model.blocks[0].current_propagate_radius == 0
    assert model.blocks[1].current_propagate_radius == 1

    model.configure_anchor_sparse_attention(
        enabled=True,
        keep_ratio=0.2,
        current_keep_ratio=0.25,
        attention_query_keep_ratio=0.25,
        dense_prefix_layers=1,
        dense_suffix_layers=0,
        propagate_radius=1,
        propagate_every=5,
        current_attention=True,
        packed_middle=True,
        probe_dim=2,
        num_router_heads=1,
        smooth_radius=0,
    )
    assert model.anchor_sparse_packed_middle
    assert model.anchor_sparse_config is not None
    assert model.anchor_sparse_config.keep_ratio == 0.2
    assert all(
        block.self_attn.anchor_sparse_config is not None
        and block.self_attn.anchor_sparse_config.keep_ratio == 1.0
        for block in model.blocks
    )
    assert model.blocks[0].self_attn.record_anchor_diagnostics
    assert not any(block.sparse_current_compute for block in model.blocks)
    assert not any(block.sparse_current_attention for block in model.blocks)
    assert model.anchor_sparse_propagate_radius == 1
    assert model.anchor_sparse_propagate_every == 5

    dynamic_table = DynamicPackedBudgetTable.constant(
        num_dit_steps=8,
        num_layers=2,
        history_keep_ratio=0.20,
        current_keep_ratio=0.25,
    )
    model.configure_dynamic_packed_budget_table(dynamic_table)
    head_group_table = DynamicPackedHeadGroupBudgetTable(
        head_keep_ratios=tuple(
            tuple((1.0, 0.20) for _ in range(2)) for _ in range(8)
        ),
    )
    model.configure_dynamic_packed_head_group_budget_table(head_group_table)
    model.set_dynamic_attention_oracle_step(
        scheduler_index=0,
        dit_index=0,
        scheduler_steps=16,
        timestep=999,
    )
    assert model._packed_budget_ratios_for_layer(1) == (0.20, 0.25)
    assert model._packed_head_groups_for_layer(1) == (
        ((0,), 1.0, None),
        ((1,), 0.20, None),
    )
    model.set_dynamic_sparse_force_dense(True)
    assert model._packed_budget_ratios_for_layer(1) == (1.0, 1.0)
    assert model._packed_head_groups_for_layer(1) is None
    model.begin_dynamic_attention_oracle_request(current_start_frame=7)
    assert model._packed_budget_ratios_for_layer(1) == (0.20, 0.25)
    route = AnchorRoute(
        video_indices=torch.arange(3 * 880).reshape(1, -1),
        scores=torch.randn(1, 3, 880),
        num_video_frames=3,
        num_dense_frames=2,
        anchor_tokens_per_sparse_frame=176,
    )
    _, _, history_indices, history_token_count = model._prepare_packed_anchor_profiles(
        route=route,
        current_frames=2,
        cache_key=("test",),
        history_keep_ratios=(0.20, 0.50),
    )
    _, _, reused_indices, _ = model._prepare_packed_anchor_profiles(
        route=route,
        current_frames=2,
        cache_key=("test",),
        history_keep_ratios=(0.20, 0.50),
    )
    assert history_token_count == 880
    assert torch.equal(
        history_indices[0.20],
        history_indices[0.50][:, : history_indices[0.20].shape[1]],
    )
    assert reused_indices[0.20] is history_indices[0.20]

    model.configure_anchor_sparse_attention(
        enabled=True,
        keep_ratio=1.0,
        current_keep_ratio=1.0,
        packed_middle=True,
    )
    assert not model.anchor_sparse_packed_middle
    assert model._dynamic_packed_budget_table is None
    assert model._dynamic_packed_head_group_budget_table is None
    assert not any(block.self_attn.record_anchor_diagnostics for block in model.blocks)

    model.configure_anchor_sparse_attention(enabled=False)
    assert model.anchor_sparse_config is None
    assert not model.anchor_sparse_packed_middle
    assert all(block.self_attn.anchor_sparse_config is None for block in model.blocks)
    assert not any(block.sparse_current_compute for block in model.blocks)
    assert not any(block.sparse_current_attention for block in model.blocks)
    assert not any(block.current_propagate_radius for block in model.blocks)


def test_sparse_current_update_keeps_unselected_video_tokens_and_updates_registers() -> None:
    module = _load_attention_module()
    x = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    e = tuple(torch.zeros(1, 7, 1, 1) for _ in range(6))
    current_video_indices = torch.tensor([[1, 3]])

    def add_ten(selected_x, selected_e):
        assert selected_x.shape == (1, 5, 1)
        assert all(part.shape == (1, 5, 1, 1) for part in selected_e)
        return selected_x + 10

    updated = module.sparse_current_token_update(
        x=x,
        e=e,
        current_video_indices=current_video_indices,
        action_register_length=3,
        anchor_sparse_config=AnchorSparseConfig(
            frame_seqlen=4,
            grid_height=2,
            grid_width=2,
            keep_ratio=0.5,
            recent_dense_frames=0,
            smooth_radius=0,
        ),
        propagate_radius=0,
        update_fn=add_ten,
    )

    assert torch.equal(updated[0, :, 0], torch.tensor([0.0, 11.0, 2.0, 13.0, 14.0, 15.0, 16.0]))
