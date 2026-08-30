import json

import torch

from groot.vla.model.dreamzero.base_vla import VLA
from groot.vla.model.dreamzero.base_vla import (
    _mark_full_checkpoint_component_loading,
)


def test_full_checkpoint_skips_redundant_component_downloads() -> None:
    action_head_cfg = {
        "_target_": "example.ActionHead",
        "config": {
            "text_encoder_cfg": {"text_encoder_pretrained_path": None},
            "image_encoder_cfg": {"image_encoder_pretrained_path": None},
            "vae_cfg": {"vae_pretrained_path": None},
            "diffusion_model_cfg": {"diffusion_model_pretrained_path": None},
        },
    }

    _mark_full_checkpoint_component_loading(action_head_cfg)

    assert action_head_cfg["config"]["skip_component_loading"] is True


def test_from_pretrained_marks_the_main_full_checkpoint_path(tmp_path) -> None:
    config = {
        "backbone_cfg": {},
        "action_head_cfg": {
            "_target_": "example.ActionHead",
            "config": {"defer_lora_injection": True},
        },
        "action_horizon": 24,
        "action_dim": 32,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    class LightweightVLA(VLA):
        def __init__(self, loaded_config):
            torch.nn.Module.__init__(self)
            self.loaded_config = loaded_config

        def load_state_dict(self, state_dict, strict=True):
            assert state_dict == {}
            return [], []

    loaded = LightweightVLA.from_pretrained(str(tmp_path))

    action_head_config = loaded.loaded_config.action_head_cfg["config"]
    assert action_head_config["skip_component_loading"] is True
    assert action_head_config["defer_lora_injection"] is False
