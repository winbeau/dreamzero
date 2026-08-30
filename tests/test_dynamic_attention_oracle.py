import torch

from groot.vla.model.dreamzero.modules.dynamic_attention_oracle import (
    DenseAttentionOracleCollector,
    DenseAttentionOracleConfig,
    OracleThresholds,
    analyze_dense_attention,
    deterministic_query_sample_indices,
    head_importance_correlation,
    minimum_oracle_budget,
    support_turnover,
)


def test_query_sampling_is_deterministic_and_covers_endpoints() -> None:
    indices = deterministic_query_sample_indices(
        11,
        4,
        device=torch.device("cpu"),
    )
    assert torch.equal(indices, torch.tensor([0, 3, 7, 10]))
    assert torch.equal(
        deterministic_query_sample_indices(3, None, device=torch.device("cpu")),
        torch.arange(3),
    )


def test_dense_attention_full_budget_is_exact_and_profiles_are_nested() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 5, 2, 4, generator=generator)
    key = torch.randn(1, 7, 2, 4, generator=generator)
    value = torch.randn(1, 7, 2, 3, generator=generator)
    statistics = analyze_dense_attention(
        query,
        key,
        value,
        keep_ratios=(1.0, 0.5, 0.25),
        query_chunk_size=2,
        support_ratio=0.5,
    )

    assert statistics["mass_mean"].shape == (3, 2)
    assert torch.allclose(statistics["mass_mean"][0], torch.ones(2), atol=1e-6)
    assert torch.allclose(
        statistics["output_cosine_mean"][0],
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.equal(
        statistics["output_relative_l2_mean"][0],
        torch.zeros(2),
    )
    assert statistics["ranked_key_indices"].shape == (2, 4)
    assert torch.all(statistics["mass_mean"][0] >= statistics["mass_mean"][1])
    assert torch.all(statistics["mass_mean"][1] >= statistics["mass_mean"][2])


def test_minimum_budget_uses_tail_quality_and_dense_fallback() -> None:
    statistics = {
        "keep_ratios": (1.0, 0.5, 0.25),
        "mass_p05": torch.tensor([[1.0, 1.0], [0.95, 0.89], [0.91, 0.80]]),
        "output_cosine_p05": torch.tensor(
            [[1.0, 1.0], [0.9995, 0.9995], [0.9992, 0.9995]]
        ),
        "output_relative_l2_p95": torch.tensor(
            [[0.0, 0.0], [0.03, 0.03], [0.04, 0.03]]
        ),
    }
    selected = minimum_oracle_budget(statistics, OracleThresholds())
    assert torch.equal(selected, torch.tensor([0.25, 1.0]))


def test_head_correlation_and_support_turnover() -> None:
    first = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    assert torch.allclose(
        head_importance_correlation(first, first),
        torch.ones(2),
    )
    previous = torch.tensor([[0, 2], [1, 3]])
    current = torch.tensor([[0, 1], [1, 3]])
    turnover = support_turnover(previous, current, num_keys=4)
    assert torch.equal(turnover, torch.tensor([0.5, 0.0]))


def test_collector_writes_one_request_with_step_layer_and_profiles(tmp_path) -> None:
    collector = DenseAttentionOracleCollector(
        DenseAttentionOracleConfig(
            output_dir=tmp_path,
            keep_ratios=(1.0, 0.5),
            max_video_queries=3,
            query_chunk_size=2,
            support_ratio=0.5,
        )
    )
    collector.begin_request(current_start_frame=7, instruction="pick object")
    collector.set_step(
        scheduler_index=6,
        dit_index=3,
        scheduler_steps=16,
        timestep=500,
    )
    generator = torch.Generator().manual_seed(11)
    collector.observe(
        layer_index=4,
        video_query=torch.randn(1, 5, 2, 4, generator=generator),
        action_query=torch.randn(1, 2, 2, 4, generator=generator),
        video_key=torch.randn(1, 6, 2, 4, generator=generator),
        video_value=torch.randn(1, 6, 2, 3, generator=generator),
    )
    outputs = collector.flush_request()
    assert outputs is not None
    jsonl_path, profiles_path = outputs
    line = jsonl_path.read_text().strip()
    assert '"scheduler_steps":16' in line
    assert '"layer_index":4' in line
    profiles = torch.load(profiles_path, weights_only=True)
    assert set(profiles) == {
        "r0_req000000_d03_l04_video",
        "r0_req000000_d03_l04_action",
    }
    assert all(profile.shape == (2, 3) for profile in profiles.values())
