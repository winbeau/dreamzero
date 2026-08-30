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
