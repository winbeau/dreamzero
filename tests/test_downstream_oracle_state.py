from types import SimpleNamespace

import pytest
import torch

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHead,
)
from socket_test_optimized_AR import ARDroidRoboarenaPolicy


def _minimal_policy_head(*, sparse: bool = False) -> WANPolicyHead:
    head = WANPolicyHead.__new__(WANPolicyHead)
    torch.nn.Module.__init__(head)
    head.model = SimpleNamespace(anchor_sparse_enabled=sparse)
    head.current_start_frame = 7
    head.language = "instruction"
    head.kv_cache1 = [torch.tensor([1.0]), torch.tensor([2.0])]
    head.kv_cache_neg = [torch.tensor([3.0])]
    head.crossattn_cache = [None, torch.tensor([4.0])]
    head.crossattn_cache_neg = [torch.tensor([5.0])]
    return head


def test_downstream_oracle_state_restore_reuses_original_cache_tensors() -> None:
    head = _minimal_policy_head()
    original_cache = tuple(head.kv_cache1)
    snapshot = head.snapshot_downstream_oracle_state()

    head.current_start_frame = 11
    head.language = "changed"
    head.kv_cache1[0] = torch.tensor([99.0])
    head.kv_cache_neg = [torch.tensor([98.0])]
    head.crossattn_cache = []
    head.crossattn_cache_neg = []
    head.restore_downstream_oracle_state(snapshot)

    assert head.current_start_frame == 7
    assert head.language == "instruction"
    assert tuple(head.kv_cache1) == original_cache
    assert head.kv_cache1[0] is original_cache[0]
    assert head.kv_cache1[1] is original_cache[1]
    assert torch.equal(head.kv_cache_neg[0], torch.tensor([3.0]))
    assert head.crossattn_cache[0] is None
    assert torch.equal(head.crossattn_cache_neg[0], torch.tensor([5.0]))


def test_downstream_oracle_state_snapshot_rejects_sparse_execution() -> None:
    head = _minimal_policy_head(sparse=True)

    with pytest.raises(RuntimeError, match="Dense path"):
        head.snapshot_downstream_oracle_state()


def test_downstream_oracle_state_restore_rejects_invalid_payload() -> None:
    head = _minimal_policy_head()

    with pytest.raises(ValueError, match="invalid"):
        head.restore_downstream_oracle_state({"current_start_frame": 0})


def test_downstream_video_return_selector_is_research_gated() -> None:
    policy = ARDroidRoboarenaPolicy.__new__(ARDroidRoboarenaPolicy)
    policy._allow_downstream_request_override = True
    observation = {"dynamic_downstream_return_video": True, "frame": 1}

    assert policy._pop_downstream_video_return(observation) is True
    assert observation == {"frame": 1}

    policy._allow_downstream_request_override = False
    with pytest.raises(ValueError, match="video return is disabled"):
        policy._pop_downstream_video_return(
            {"dynamic_downstream_return_video": True}
        )


def test_downstream_video_return_selector_rejects_non_boolean() -> None:
    policy = ARDroidRoboarenaPolicy.__new__(ARDroidRoboarenaPolicy)
    policy._allow_downstream_request_override = True

    with pytest.raises(ValueError, match="must be boolean"):
        policy._pop_downstream_video_return(
            {"dynamic_downstream_return_video": 1}
        )
