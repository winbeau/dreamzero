# DreamZero (RLinf WAN2.2-5B LIBERO-SFT) → action-eval inference path

Status: **static code investigation only.** Nothing in this document was executed. No
simulator, no checkpoint, no GPU, no remote host was touched. Every claim is labelled:

- **[V]** verified by reading the cited file and line in a local checkout.
- **[I]** inferred from code structure; the mechanism is clear but the exact artifact
  was not available to confirm.
- **[U]** unverifiable locally: it depends on the downloaded checkpoint or on a GPU run.

Local sources that were read:

- `/home/winbeau/Papers/ICLR2027-WAM-SA/RLinf` (branch `main`, `a5ccaf59`)
- `/home/winbeau/Papers/ICLR2027-WAM-SA/dreamzero` (branch `main`, `e193cfd`)
- `/home/winbeau/Papers/ICLR2027-WAM-SA/action-eval`
- Checkpoint `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000` is **not** on this
  machine; there is no local `metadata.json` copy anywhere in the three repos **[V]**
  (`find` over the workspace returned only an unrelated webencodings file).

---

## Summary of the recommended path

Run the adapter in the RLinf embodied virtualenv and drive RLinf's own
`DreamZeroPolicy` — do not reimplement preprocessing. Concretely:

1. Build an OmegaConf config from RLinf's committed model architecture
   (`RLinf/examples/embodiment/config/model/dreamzero_5b.yaml` **[V]**), set
   `model_path` to the checkpoint directory, set `metadata_json_path` explicitly, set
   `tokenizer_path` to a reachable tokenizer, `precision: bf16`, `is_lora: False`,
   `embodiment_tag: libero_sim`, `action_horizon / num_action_per_block /
   num_action_chunks: 16`, `target_video_height: 160`, `target_video_width: 320`
   **[V]** (all values taken from `libero_spatial_dreamzero_eval.yaml` and from the
   checkpoint `config.json` as reported in the task).
2. Run RLinf's own merge/validation helper `validate_dreamzero_sft_model_cfg` so the
   checkpoint's `config.json` is folded in with RLinf's documented precedence rules
   (`dreamzero_config.py:161-211` **[V]**), then call
   `rlinf.models.embodiment.dreamzero.get_model(cfg, torch.bfloat16)` **[V]**
   (`RLinf/rlinf/models/embodiment/dreamzero/__init__.py:76-205`). This constructs
   `DreamZeroPolicy` (a `groot.vla.model.dreamzero.base_vla.VLA` subclass) and loads
   `model.safetensors.index.json` + the six shards.
3. Per observation, call
   `policy.predict_action_batch(env_obs, mode="eval")` with
   `env_obs = {"main_images": agentview[None], "wrist_images": wrist[None],
   "states": state[None].astype(np.float32), "task_descriptions": [instruction]}` —
   the images are already flipped by the benchmark and must **not** be flipped again
   **[V]** (`action-eval/src/action_eval/benchmarks/libero/backend.py:87-112` vs
   `RLinf/rlinf/envs/sim/libero/utils.py:79-109`). Return `actions[0]` (shape `[16,7]`).

The load-bearing risks, in order: (a) the checkpoint's shards may contain
FSDP-promoted scalar parameters of shape `[1]` that RLinf's own loader cannot load into
a fresh `[]`-shaped model, because `_promote_scalar_params_to_1d` runs *after*
`load_state_dict` (§1.5, **[I]**); (b) the checkpoint's `config.json` carries absolute
training-machine paths that win RLinf's path merge and must be overridden (§1.3,
**[V]**); (c) the 160×320 vs 176×320 target-video resolution silently changes the DiT
token grid (§2.6, **[V]**); (d) forgetting the q99 action denormalisation is silent
(§7).

---

## 1. The exact load path

### 1.1 What RLinf actually calls

The entry point is `get_model` in
`RLinf/rlinf/models/embodiment/dreamzero/__init__.py`. It is reached in production from
`RLinf/rlinf/workers/rollout/hf/huggingface_worker.py:147`
(`self.hf_model: BasePolicy = get_model(rollout_model_config)`) **[V]**, which goes
through the dispatcher `rlinf/models/__init__.py:359-373`:

```python
def get_model(cfg: DictConfig):
    model_type = str(cfg.model_type)
    model_builder = _MODEL_REGISTRY.get(model_type)
    if model_builder is None:
        return None
    torch_dtype = torch_dtype_from_precision(cfg.precision)
    model = model_builder(cfg, torch_dtype)
    if (Worker.torch_platform is not None and Worker.torch_platform.is_available()
            and cfg.get("load_to_device", True)):
        model = model.to(Worker.torch_device_type)
    if cfg.is_lora:
        ...
```

So the dispatcher requires the keys `model_type`, `precision`, `is_lora` and (unless
`load_to_device: False`) that `Worker.torch_platform` be usable; it does **not** require
a Ray cluster (only that `ray` is importable — see §3) **[V]**.

`dreamzero/__init__.py:76-205` then does, in order **[V]**:

```python
def get_model(cfg: DictConfig, torch_dtype=None):
    from rlinf.utils.patcher import Patcher
    _configure_torch_dynamo_for_dreamzero_inference()
    Patcher.clear()
    Patcher.add_patch("groot...wan_video_vae.WanVideoVAE", "rlinf...patch.wan_video_vae.WanVideoVAE")
    Patcher.add_patch("groot...wan_video_vae.WanVideoVAE38", "rlinf...patch.wan_video_vae.WanVideoVAE38")
    Patcher.add_patch("groot...wan_video_vae.WanVideoVAEStateDictConverter", "rlinf...patch.wan_video_vae.WanVideoVAEStateDictConverter")
    _dit_chunk = "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk"
    Patcher.add_wrapper(f"{_dit_chunk}.CausalWanSelfAttention._process_clean_image_only", torch.compile(mode="reduce-overhead"))
    Patcher.add_wrapper(f"{_dit_chunk}.CausalWanSelfAttention._process_state_blocks", torch.compile(mode="reduce-overhead"))
    Patcher.add_wrapper(f"{_dit_chunk}.CausalWanSelfAttention._process_noisy_image_blocks", torch.compile(mode="reduce-overhead"))
    Patcher.add_wrapper(f"{_dit_chunk}.CausalWanSelfAttention._process_noisy_action_blocks", torch.compile(mode="reduce-overhead"))
    Patcher.add_patch(f"{_dit_chunk}.CausalWanModel._forward_train", "rlinf...patch.wan_causal_model_forward_train._forward_train")
    Patcher.add_patch("groot.vla.data.schema.embodiment_tags.EmbodimentTag", "rlinf...data_transforms.embodiment_tag.EmbodimentTag")
    Patcher.apply()

    model_path = cfg.get("model_path", None)
    tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    dreamzero_config = DreamZeroConfig(**config_dict)

    has_full_model_weights = False
    if model_path is not None:
        ckpt = Path(model_path)
        st = ckpt / "model.safetensors"
        st_index = ckpt / "model.safetensors.index.json"
        has_full_model_weights = st.exists() or st_index.exists()

    dreamzero_config.action_head_cfg["config"]["defer_lora_injection"] = False
    dreamzero_config.action_head_cfg["config"]["skip_component_loading"] = has_full_model_weights

    metadata = load_dreamzero_dataset_metadata(cfg)
    data_transforms = build_dreamzero_composed_transform(cfg, tokenizer_path)
    data_transforms.set_metadata(metadata)
    data_transforms.eval()

    embodiment_tag = str(cfg.embodiment_tag)
    dreamzero_config.data_transforms = data_transforms
    dreamzero_config.embodiment_tag = embodiment_tag
    dreamzero_config.relative_action = bool(cfg.get("relative_action", False))
    dreamzero_config.relative_action_per_horizon = bool(cfg.get("relative_action_per_horizon", False))
    dreamzero_config.relative_action_keys = list(cfg.get("relative_action_keys") or [])

    model = DreamZeroPolicy(config=dreamzero_config)
    if has_full_model_weights and model_path is not None:
        state_dict = {}
        if st_index is not None and st_index.exists():
            with open(st_index, "r") as f: index = json.load(f)
            for shard_file in sorted(set(index["weight_map"].values())):
                state_dict.update(load_file(str(ckpt / shard_file)))
        elif st is not None and st.exists():
            state_dict.update(load_file(str(st)))
        if any(".base_layer." in k for k in state_dict):
            state_dict = {k.replace(".base_layer.", "."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        get_logger().warning("No model.safetensors under %s; initializing DreamZero from component weights ...", loc)

    if hasattr(model, "action_head"):
        ah = model.action_head
        if not hasattr(ah, "trt_engine"): ah.trt_engine = None
        if not hasattr(ah, "trt_context"): ah.trt_context = None
    _promote_scalar_params_to_1d(model)
    model = model.to(dtype=torch_dtype)
    return model
```

Decisive facts:

- **It does not use `VLA.from_pretrained`.** The RLinf path builds the model from the
  Hydra config object and loads a plain `state_dict` afterwards **[V]**. The
  `from_pretrained` classmethod in `dreamzero/groot/vla/model/dreamzero/base_vla.py:511-589`
  is *not* on this path; it is only reachable if an adapter calls it directly (that is
  the §6 alternative).
- **`skip_component_loading` is set from file existence**, so with the six shards plus
  `model.safetensors.index.json` present the external WAN/T5/CLIP weights are *not*
  downloaded **[V]** (see §5). If the index file were absent, RLinf would **silently**
  build the generic DreamZero base from HF components instead of the LIBERO-SFT weights
  and only emit a warning **[V]** — this must be checked on the checkpoint.
- **`DataParallel`/FSDP is not involved**; there is no `dist.init_process_group`
  anywhere on this path (`VLA.__init__` only does
  `self.rank = dist.get_rank() if dist.is_initialized() else 0`,
  `base_vla.py:79`) **[V]**.

### 1.2 The class and its config object

- **Class:** `rlinf.models.embodiment.dreamzero.dreamzero_policy.DreamZeroPolicy`
  (`dreamzero_policy.py:32`), declared as
  `class DreamZeroPolicy(VLA, BasePolicy)` where
  `VLA` is `groot.vla.model.dreamzero.base_vla.VLA` (`dreamzero_policy.py:20,32`) **[V]**.
  Its `__init__` only sets `self.config`, resolves the rollout observation layout and the
  action key list (`dreamzero_policy.py:44-59`); the real construction happens in
  `VLA.__init__` (`base_vla.py:66-79`), which does
  `self.backbone = instantiate(config.backbone_cfg)` and
  `self.action_head = instantiate(config.action_head_cfg)` via `hydra.utils.instantiate`
  **[V]**.
- **Config object:** `DreamZeroConfig` (`dreamzero_config.py:214-255`), a `dataclass`
  subclassing `groot...base_vla.VLAConfig` (itself a `transformers.PretrainedConfig`)
  with extra fields `backbone_cfg`, `action_head_cfg`, `action_horizon`, `action_dim`,
  `num_action_chunks`, `relative_action`, `relative_action_per_horizon`,
  `relative_action_keys`, `data_transforms`, `embodiment_tag`
  (`dreamzero_config.py:215-250`) **[V]**. `__init__` sets every `kwargs` entry as an
  attribute, so unknown `config.json` keys are tolerated **[V]**.
- **Constructor arguments:** only `config=` (`dreamzero_policy.py:44-48`). No device, no
  dtype, no checkpoint path.

### 1.3 Required vs optional config keys

Required for `dreamzero/__init__.py:get_model` to reach a working policy **[V]**:

| key | why |
|---|---|
| `model_type: dreamzero` | dispatcher lookup (`models/__init__.py:360`) |
| `precision` | `torch_dtype_from_precision` (`models/__init__.py:365`) |
| `is_lora` | `if cfg.is_lora:` (`models/__init__.py:375`) |
| `embodiment_tag: libero_sim` | `str(cfg.embodiment_tag)` (`__init__.py:159`) and `rollout_obs_layout_for_embodiment` (`dreamzero_policy.py:55`) |
| `model_path` | decides full-weight loading (`__init__.py:123-141`) |
| `tokenizer_path` (non-null!) | see below |
| `action_head_cfg` (with `_target_`, `config.diffusion_model_cfg.max_chunk_size`, `config.num_frames`, `config.{text_encoder,image_encoder,vae}_cfg`) | `instantiate` in `VLA.__init__`, `WANPolicyHead.__init__` (`wan_flow_matching_action_tf.py:174-176, 237`) |
| `backbone_cfg` with `_target_: ...IdentityBackbone` | `assert isinstance(config.backbone_cfg, dict)` (`base_vla.py:70`) |
| `action_horizon`, `action_dim`, `max_action_dim`, `max_state_dim`, `state_horizon`, `target_video_height/width`, `num_action_per_block` | read by the transforms and by `WANPolicyHeadConfig` |
| `metadata_json_path` or `model_path/experiment_cfg/metadata.json` | `load_dreamzero_dataset_metadata` (`data_transforms/__init__.py:153-174`) |

Two traps that produce errors rather than silent degradation, but are easy to hit:

1. `tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")`
   (`__init__.py:125`) uses the dictionary default **only when the key is absent**. The
   committed model YAML sets `tokenizer_path: null` **[V]**
   (`examples/embodiment/config/model/dreamzero_5b.yaml:4`), so a null value propagates
   and the lazy tokenizer construction eventually fails in
   `AutoTokenizer.from_pretrained(None)` **[I]**. The adapter must set a real value.
2. `metadata_json_path` is **not** in `_PATH_OVERRIDE_KEYS`
   (`dreamzero_config.py:29-37`: only `tokenizer_path` and the four
   `*_pretrained_path`s). For any other key, `_yaml_takes_precedence` returns
   `yaml_val is not None` (`dreamzero_config.py:48-51`), so a **null YAML value loses to
   the checkpoint value**. The checkpoint's `metadata_json_path` is an absolute
   training-machine path. Under the merge helper, therefore, an explicit non-null
   `metadata_json_path` is the only way to be sure. (For `tokenizer_path`, the opposite
   holds: a non-empty YAML value wins over the checkpoint, which is what we want
   **[V]**.) Conversely, if the adapter builds the config by plain
   `OmegaConf.merge(checkpoint_cfg, yaml)`, YAML-null would win and the fallback
   `model_path/experiment_cfg/metadata.json` would be used.

`validate_dreamzero_sft_model_cfg` (`dreamzero_config.py:161-211`) is the function that
implements this merge, logs field-level mismatches, and asserts `embodiment_tag`,
`tokenizer_path`, `action_head_cfg.config.diffusion_model_cfg.max_chunk_size`,
`config.num_frames` and `action_horizon` **[V]**. It is called from RLinf's SFT config
validation (`rlinf/config.py:1423-1427`) and from the checkpoint converter
(`utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py:180-186`) **[V]**. Reusing it is
the most faithful way to reproduce "checkpoint `config.json` + YAML architecture".

### 1.4 What is in `patch/`

`RLinf/rlinf/models/embodiment/dreamzero/patch/` contains two modules, registered in
`get_model` via RLinf's `Patcher` (`rlinf/utils/patcher.py`) **[V]**:

- `patch/wan_video_vae.py` — re-implements `WanVideoVAE` / `WanVideoVAE38` /
  `WanVideoVAEStateDictConverter`. The documented purpose is batched VAE encode for
  micro-batch > 1 during training (`patch/wan_video_vae.py:26-29`) **[V]**. Side effects
  that matter for a single-process adapter: the constructor puts `mean`/`std` on
  `device="cuda"` unconditionally (`patch/wan_video_vae.py:44-48`; same in the upstream
  `groot/vla/model/dreamzero/modules/wan_video_vae.py:1053-1054,1369-1371`) **[V]**, so
  model construction requires a usable CUDA device; and `encode` accepts
  `(videos, tiled=False, ...)` (`patch/wan_video_vae.py:247-252`) **[V]**.
- `patch/wan_causal_model_forward_train.py` — replaces
  `CausalWanModel._forward_train` to disable gradient checkpointing for micro-batch > 1
  (`patch/wan_causal_model_forward_train.py:17-19`) **[V]**. `CausalWanModel.forward`
  dispatches on `kv_cache` (`wan_video_dit_action_casual_chunk.py:2635-2643`), and the
  inference path always passes `kv_cache=` (`wan_flow_matching_action_tf.py:897,908`), so
  this patch is **training-only** and never runs at inference **[V]**.
- The four `torch.compile(mode="reduce-overhead")` wrappers added in the same
  `get_model` call **do** apply to inference: `_process_clean_image_only`,
  `_process_state_blocks`, `_process_noisy_image_blocks`, `_process_noisy_action_blocks`
  are called from `CausalWanAttentionBlock.forward`
  (`wan_video_dit_action_casual_chunk.py:1012-1032`) which is reached through
  `_forward_inference` (`:2376`) **[V]**. Consequence: the **first** `predict` pays
  Dynamo compilation plus CUDA-graph capture; `_configure_torch_dynamo_for_dreamzero_inference`
  (`__init__.py:30-54`) raises the recompile limits for exactly this reason **[V]**. There
  is no env switch in the RLinf path that disables those wrappers (the
  `DREAMZERO_DISABLE_TORCH_COMPILE` switch only guards `WANPolicyHead.post_initialize`,
  `wan_flow_matching_action_tf.py:1374-1393`, which RLinf never calls) **[V]**.
- The `EmbodimentTag` patch (`__init__.py:113-116`) replaces groot's enum with RLinf's
  three-member enum (`data_transforms/embodiment_tag.py:24-29`) **[V]**. It matters
  because `DreamTransform.apply_single` (`data_transforms/dream_transform.py:86-193`)
  avoids groot branches that reference enum members RLinf removed **[V]**, and because
  the upstream `collate` in `groot/.../dreamzero_cotrain.py:92-164` would fail for
  `libero_sim` (see §6) **[V]**.

### 1.5 The one construction detail that can break the load

`_promote_scalar_params_to_1d(model)` is called **after** `model.load_state_dict(...)`
(`__init__.py:188` then `:202`) **[V]**. The checkpoint shards were written by
`save_state_dict_sharded_safetensors` from `model.state_dict()` of a model built by this
same `get_model` (`utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py:347,328-333`)
**[V]**, i.e. from an already-promoted model. groot's image encoder contains genuine 0-d
parameters — `CLIP.log_scale` / `XLMRobertaCLIP.log_scale`
(`groot/vla/model/dreamzero/modules/wan_video_image_encoder.py:568,712`), reached from
`WanImageEncoder.__init__` via `clip_xlm_roberta_vit_h_14`
(`wan_video_image_encoder.py:856-870`) **[V]** — so the shards plausibly carry
`action_head.image_encoder.model.log_scale` with shape `[1]` while a freshly constructed
model has shape `[]` **[I]**. `torch.nn.Module.load_state_dict(strict=False)` tolerates
missing/unexpected keys but **still raises** on a size mismatch **[I]**, so the
`model_path=<dir>` route may fail with "size mismatch" for exactly these keys; the
converter itself loads the training `.pt` after promoting, so the training artifact and
the HF artifact are consistent with each other but not with the fresh-model load order.

This is consistent with the committed eval YAML, which avoids the issue entirely by
setting `model_path: null` and loading a `.pt` through `runner.ckpt_path`
(`evaluations/libero/libero_spatial_dreamzero_eval.yaml`, and
`huggingface_worker.py:152-154`) **[V]** — but a `.pt` is not what the HF repo ships.

**Preflight that settles it without a GPU:** read the safetensors header of one shard
(JSON in the first 8 bytes) and look for `log_scale` / `logit_bias` with shape `[1]`. If
present, the adapter must replicate `get_model`'s body and promote **before** loading
(see the skeleton in the last section), or `assign=True`/cast the incoming tensors.

---

## 2. The exact inference call

### 2.1 Observation → modality keys

`DreamZeroPolicy.predict_action_batch(env_obs, mode, **kwargs)`
(`dreamzero_policy.py:261-305`) **[V]**:

```python
converted_obs = self._observation_convert(env_obs)     # -> _observation_convert -> convert_rollout_env_obs
batch = Batch(obs=converted_obs)
normalized_input = self._process_batch(batch)
with torch.no_grad():
    model_pred = self.lazy_joint_video_action_causal(normalized_input)
normalized_action = model_pred["action_pred"].float()
batch = self.unapply(Batch(normalized_action=normalized_action), obs=converted_obs)
actions = self._actions_from_unapply(batch.act)
if self._rollout_obs_layout.binarize_gripper:
    actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(actions.dtype)
```

`_observation_convert` → `convert_rollout_env_obs(embodiment_tag, env_obs)`
(`dreamzero_policy.py:240-242`, `data_transforms/__init__.py:125-134`) uses
`LiberoSimDataTransform.ROLLOUT_OBS_LAYOUT` (`libero_sim.py:58-65`) **[V]**:

```python
video_fields=(("main_images", "video.image"), ("wrist_images", "video.wrist_image")),
state_fields=(("states", "state.state"),),
state_fields... binarize_gripper=True,
language_env_key="task_descriptions",
```

and `convert_rollout_env_obs_with_layout` (`data_transforms/base.py:141-186`) **[V]**:

- `images` are normalised to `[B,T,H,W,C]` uint8 by `_rollout_to_bthwc_uint8`
  (`base.py:73-100`). **No flip is performed** — it only handles channel-order and
  dtype (`if arr.max() <= 1.0: arr = (arr*255)...`).
- `states` become float32 and are inserted as `states[:, None, :]`
  (`base.py:156-165`).
- the instruction lands in `annotation.task` as a `list[str]`
  (`base.py:172-177`); `task_descriptions` may be a bare string and is broadcast.

Order matters: index 0 of the concatenated video becomes the **left** view
("the left view shows the exterior camera and the right view shows the wrist camera",
`libero_sim.py:46-49`) **[V]**. `main_images` (agentview) must stay first —
swapping agentview/wrist silently inverts the prompt and the learned view layout.

### 2.2 The transform chain (`data_transforms.py`)

The file named in the task, `rlinf/data/datasets/dreamzero/data_transforms.py`, is now a
package `data_transforms/` **[V]**. Three functions matter:

- `convert_rollout_env_obs` / `convert_rollout_env_obs_with_layout`
  (`data_transforms/__init__.py:125-134`, `base.py:141-186`) — above.
- `rollout_obs_layout_for_embodiment` (`data_transforms/__init__.py:120-122`) — returns
  the static layout described above.
- `collect_dreamzero_dataset_keys` (`data_transforms/__init__.py:137-150`) — walks
  `data_transform.transforms` collecting `video_concat_order` / `state_concat_order` /
  `action_concat_order`; for libero this yields
  `(["video.image","video.wrist_image"], ["state.state"], ["action.actions"], ["annotation.task"])` **[V]**,
  and `DreamZeroPolicy` keeps only the action keys (`dreamzero_policy.py:56-59`) so
  `_actions_from_unapply` concatenates `action.actions` (`:244-259`).

`build_dreamzero_composed_transform` (`data_transforms/__init__.py:194-211`) builds the
chain from `LiberoSimDataTransform.get_transform` / `_build_composed_transform`
(`libero_sim.py:112-212`) **[V]**. In order:

| # | transform | effect at inference (`.eval()`) |
|---|---|---|
| 0 | `VideoToTensor(apply_to=["video.image","video.wrist_image"], backend="torchvision")` | uint8 `[b,t,h,w,c]` → float `[n,c,h,w]` in `[0,1]` (`video.py:534-594`); note the base `apply` rearranges with a mismatched-looking pattern that is in fact label-agnostic (`video.py:156-171`) |
| 1 | `VideoCrop(scale=0.95)` | **eval mode = `T.CenterCrop((int(H*0.95), int(W*0.95)))`** (`video.py:237-263`); `H,W` come from `metadata.modalities.video.<key>.resolution` (`video.py:250-256`) and `check_input` **asserts the incoming frame equals that resolution** (`video.py:276-288`). Note the unpack `self.width, self.height = self.original_resolutions[...]` (`video.py:252`) is swapped and `check_input` re-swaps, so the whole stage is only self-consistent (and non-square metadata only correct) when `H == W` — the benchmark renders 256×256, so this holds |
| 2 | `VideoResize(height=256, width=256, interpolation="linear")` | hard-coded 256×256 — `get_transform` never passes `target_video_height/width` (`libero_sim.py:112-139` vs the defaults at `:153-154`) |
| 3 | `VideoColorJitter` | eval mode returns `None` → no-op (`video.py:139-145,459`) |
| 4 | `VideoToNumpy` | back to uint8 `[...h,w,c]` (`video.py:597-622`) |
| 5 | `StateActionToTensor(apply_to=["state.state"])` | numpy → tensor |
| 6 | `StateActionTransform(apply_to=["state.state"], normalization_modes={"state.state":"q99"})` | q99 normalisation |
| 7 | `StateActionToTensor(apply_to=["action.actions"])` | present for symmetry; unused at inference |
| 8 | `StateActionTransform(...,{"action.actions":"q99"})` | used **only** in reverse (`unapply`) |
| 9 | `ConcatTransform` | packs `video` to `[T,V,H,W,C]`, `state` to `[T,D_state]`, `action` to `[T,D_action]` (`groot/vla/data/transform/concat.py:86-171`) |
| 10 | `RLinf DreamTransform` | multi-view concat, prompt, pads to `max_*_dim`, and (via `apply_batch`) tokenises |

Chained along RLinf's subclass:

- `DreamTransform.__init__` deliberately skips groot's eager tokenizer construction and
  builds it lazily (`data_transforms/dream_transform.py:68-84`) **[V]** — but the
  tokenizer *is* still constructed during inference because `apply_batch` tokenises
  (below).
- `DreamTransform.apply_single` (`dream_transform.py:86-193`) calls `_prepare_video`,
  `_prepare_language`, `_apply_vlm_processing`, `_prepare_state`; it writes
  `action`/`action_mask` **only when `self.training`** (`:109-122`) **[V]** — so the
  inference observation carries no ground-truth action. It also always emits the fixed
  `text_negative` string (`:124-131`), used as the CFG-unconditional prompt.
- `DreamTransform._prepare_video` (`dream_transform.py:213-221`) rearranges
  `t v h w c -> v t c h w` then calls the embodiment's `concat_multiview_video`, which for
  LIBERO is a **horizontal** concat, exterior left | wrist right
  (`libero_sim.py:72-83`) **[V]**. (groot's own `_prepare_video` would build a 2×2 grid —
  `dreamzero_cotrain.py:357-379` — which is *not* what this checkpoint was trained with.)
- `DreamTransform.apply` is groot's (`dreamzero_cotrain.py:615-622`): in eval with a
  5-D video it adds a batch dim and routes to `apply_batch` **[V]**. RLinf overrides
  `apply_batch` (`dream_transform.py:195-211`) to call
  `DreamZeroCollator.collate_batch` (`data/datasets/dreamzero/dataloader.py:54-88`)
  **[V]**, which is where tokenisation and prompt wrapping happen.

### 2.3 Prompt / instruction

`collate_batch` (`dataloader.py:61-73`) **[V]**:

```python
if key == "text":
    texts = [format_training_prompt(normalize_instruction_text(elem[key]),
                                    int(elem["embodiment_id"]), embodiment_tag_mapping) for elem in features]
    ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
```

- `normalize_instruction_text` (`data_transforms/__init__.py:81-91`) runs
  `ast.literal_eval` when possible and **lowercases** the result **[V]**.
- `format_training_prompt` for LIBERO (`libero_sim.py:67-70`) is:

  `"A multi-view video shows that a robot " + instruction + " The video is split into two horizontal views: the left view shows the exterior camera and the right view shows the wrist camera. The robot " + instruction`

  (exactly `_TRAINING_PROMPT_PREFIX` + `instruction` + `_MULTIVIEW_LAYOUT` + `instruction`)
  **[V]**. The adapter must pass the **bare** benchmark instruction; wrapping it a second
  time would silently change the prompt.
- `embodiment_id` = `self.embodiment_tag_mapping["libero_sim"]` = **21**
  (`libero_sim.py:56`, `dreamzero_cotrain.py:244-249`) **[V]**.
- The tokenizer is `HuggingfaceTokenizer(tokenizer_path, seq_len=512, clean="whitespace")`
  (`dreamzero_cotrain.py:38-53`, `dream_transform.py:76-84`) **[V]**, so `tokenizer_path`
  is a genuine inference-time dependency (§5).
- `text_negative` is tokenised to `text_attention_mask_negative` (`dataloader.py:74-78`)
  and is used as the unconditional branch because `cfg_scale = 5.0`
  (`wan_flow_matching_action_tf.py:182`, `_prepare_text_inputs` at `:849-861`) **[V]**.

### 2.4 State normalisation

`StateActionTransform` with `normalization_modes={"state.state": "q99"}`
(`libero_sim.py:182-186`) uses `Normalizer.forward` (`state_action.py:99-121`) **[V]**:

```python
q01 = self.statistics["q01"]; q99 = self.statistics["q99"]
normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask])
normalized[..., mask] = 2 * normalized[..., mask] - 1
normalized = torch.clamp(normalized, -1, 1)
```

The statistics come from `metadata.statistics.state.state` via
`StateActionTransform.set_metadata` (`state_action.py:673-758`, notably
`:698-703`) **[V]**, which requires `q01` and `q99` to be present
(`DatasetStatisticalValues` declares all six fields as required,
`groot/vla/data/schema/lerobot.py:217-223`) **[V]**. There is no rotation transform for
LIBERO (`target_rotations` is empty), and `_input_dtypes` is only populated by `apply`,
so at inference the state dtype stays float32 → cast later
(`dreamzero_policy.py:229-237`) **[V]**.

Then `DreamTransform._prepare_state` (`dreamzero_cotrain.py:442-473`) pads the 8-dim
state to `max_state_dim = 64` and produces `state_mask` marking the first 8 dims
**[V]**.

### 2.5 The model call

`lazy_joint_video_action_causal` is `VLA.lazy_joint_video_action_causal`
(`base_vla.py:192-201`) **[V]**:

```python
def lazy_joint_video_action_causal(self, inputs, latent_video=None):
    backbone_inputs, action_inputs = self.prepare_input(inputs)
    backbone_outputs = self.backbone(backbone_inputs)          # IdentityBackbone -> empty features
    action_head_outputs = self.action_head.lazy_joint_video_action(backbone_outputs, action_inputs, latent_video=latent_video)
    self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
    return action_head_outputs
```

`prepare_input` (`base_vla.py:254-269`) casts floating tensors to `self.action_head.dtype`
on `self.device`; `validate_data` (`:117-149`) asserts the returned
`action_pred` has shape `[B, action_horizon, action_dim]` — so a config mismatch here is a
loud error, not a silent one **[V]**.

Inside `WANPolicyHead.lazy_joint_video_action` (`wan_flow_matching_action_tf.py:985-1350`)
**[V]**:

- `videos` = `data["images"]`, shape `[B,T,H,W,C]` with `T=1` (single rollout frame);
  `rearrange("b t h w c -> b c t h w")` (`:1003-1008`); uint8 → `/255` → `Normalize(0.5,0.5)`
  → `[-1,1]` (`:1010-1019`).
- **Target resize** (`:1024-1041`): if `self.config.target_video_height/width` are set,
  bilinear-interpolate to exactly that size. Otherwise it falls back to **176×320** when
  `model.frame_seqlen in (50,55)` and to no resize otherwise **[V]**. Note the data
  pipeline already resized to 256×256 (per view) and concatenated to 256×512, so this
  second resize is the one that produces the DiT token grid.
- `state_features = state.to(bfloat16)` (`:1021`).
- Prompt encoding with the text encoder, both conditional and unconditional
  (`:1063-1064, 849-861`).
- `encode_image(image, num_frames, height, width)` (`:575-593`) runs the CLIP image encoder
  (`self.image_encoder.encode_image`) and VAE-encodes the first frame plus zeros to build
  `ys`; `_ensure_vae_on_device` lazily moves the VAE to the input's device and bf16
  (`:562-567`) **[V]**.
- `noise_obs` and `noise_action` are drawn with `self.seed = 1140` through
  `generate_noise(..., seed=self.seed)` (`:181, 827-830, 1116-1117`) → **deterministic
  per call**; `rng_mode` must be `fixed_per_predict`.
- Denoising: `FlowUniPCMultistepScheduler` for video and action with
  `self.num_inference_steps = 16` (`:180, 1211-1222`) and shift `self.sigma_shift = 5.0`
  (`:184`); the number of actual DiT evaluations is controlled by
  `self.dit_step_mask`, 8-of-16 by default (`:210-223`) and by `should_run_model`
  (`:955-983`, disabled dynamic schedule by default) **[V]**.
- `cfg_scale = 5.0` (`:182`, `:1289`).
- **Action chunk length = `self.action_horizon`**, i.e. the `action_horizon` config key
  (16 for the LIBERO checkpoint): `noise_action` is
  `(B, self.action_horizon, self.model.action_dim)` (`:1117`) and the return is
  `{"action_pred": latents_action, "video_pred": ...}` with `action_pred` of shape
  `(B, 16, 32)` (`:1350`) **[V]**.
- Because the observation always has `T == 1`, the branch at `:1051-1053`
  (`elif videos.shape[2] == 1: self.current_start_frame = 0`) fires on every call
  **[V]**, so each `predict` re-encodes the image, rebuilds the KV/cross-attention caches
  (`:1128-1140`) and is effectively stateless. `self.language`, `clip_feas`, `ys`,
  `current_start_frame` nevertheless persist on the object; `reset()` should clear them.

### 2.6 Denormalisation back to simulator units

`DreamZeroPolicy.unapply` (`dreamzero_policy.py:120-219`) **[V]**:

```python
unnormalized_action = self.config.data_transforms.unapply({"action": batch.normalized_action.cpu()})
relative_action = self.config.relative_action
...  # only if relative_action or relative_action_per_horizon and relative_action_keys and obs is not None
batch.act = unnormalized_action
```

- `ComposedModalityTransform.unapply` walks the chain in reverse
  (`groot/vla/data/transform/base.py:138-146`) **[V]**:
  `DreamTransform.unapply` (pass-through, `dreamzero_cotrain.py:624-626`) →
  `ConcatTransform.unapply`, which slices `[..., 0:7]` using
  `action_dims["action.actions"]` taken from metadata (`concat.py:173-193`,
  `:250-258`) → `StateActionTransform.unapply`, which applies the q99 **inverse**
  `(x+1)/2*(q99-q01)+q01` (`state_action.py:188-195`) **[V]**.
- `_actions_from_unapply` concatenates the action keys in order
  (`dreamzero_policy.py:244-259`) → `(B,16,7)`.
- **The gripper dimension is transformed**: it is one of the 7 columns, so it goes
  through the same q99 inverse, and then `predict_action_batch:289-292` **binarises** it:
  `actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0)` because
  `LiberoSimDataTransform.ROLLOUT_OBS_LAYOUT.binarize_gripper = True`
  (`libero_sim.py:64`) **[V]**. An adapter that skips this returns a continuous gripper
  command; LIBERO's OSC controller thresholds it anyway, but the values and the
  resulting contact behaviour are not the validated ones.
- `relative_action` is `False` for this checkpoint, and `relative_action_keys` defaults
  to `[]` (`__init__.py:166-169`), so the state-addition block at
  `dreamzero_policy.py:150-216` is skipped entirely **[V]** (see §7).

Verified input/output contract for the adapter **[V]**:

- input: `env_obs` with `main_images` `[B,H,W,3]` uint8, `wrist_images` `[B,H,W,3]` uint8,
  `states` `[B,8]` float32, `task_descriptions` `list[str]` (or a bare string);
- output: `actions` `np.ndarray[B,16,7]` float32 (and a `result` dict with
  `prev_logprobs` / `prev_values` / `forward_inputs` that the adapter can ignore).

---

## 3. Can it run outside Ray/hydra?

**Yes for Ray, mostly yes for hydra — but not without importing both packages, and not
without importing most of RLinf.** Details **[V]**:

- **Ray:** no `ray.init`, no Ray actor, no placement, no distributed process group is
  needed. `VLA.__init__` guards the only distributed call
  (`self.rank = dist.get_rank() if dist.is_initialized() else 0`, `base_vla.py:79`) and
  the DiT checks `if not dist.is_initialized() or dist.get_rank() == 0`
  (`wan_video_dit_action_casual_chunk.py:1944,2102`). **However**, `ray` must be
  *importable*: `dreamzero_config.py:27` calls `get_logger()` at module import, and
  `rlinf/utils/logging.py:19-23` imports `rlinf.scheduler.worker.Worker`, which does
  `import ray` / `import ray.util.state` (`rlinf/scheduler/worker/worker.py:29-30`)
  **[V]**. Importing `rlinf.models.embodiment.dreamzero` additionally executes
  `rlinf/__init__.py` (omegaconf resolver), `rlinf/models/__init__.py` →
  `rlinf.config` → `rlinf.envs`, `rlinf.robotics.robots`, `rlinf.scheduler.cluster`
  (`rlinf/config.py:28-38`) **[V]**. Expense, not Ray runtime.
- **Hydra:** `hydra-core` must be importable because `VLA.__init__` calls
  `hydra.utils.instantiate` on the nested `_target_` configs
  (`base_vla.py:4,73-74`; also `wan_flow_matching_action_tf.py:10,174-176,237`)
  **[V]**. No `@hydra.main`, no config search path, no Hydra job is required — a plain
  `dict`/`DictConfig` with `_target_` keys is enough, and RLinf ships the whole
  architecture graph in one YAML (`examples/embodiment/config/model/dreamzero_5b.yaml`)
  **[V]**.
- **FSDP/Megatron:** not required. `get_model` never wraps the model; FSDP is applied by
  the actor worker, not by the rollout worker **[V]**
  (`rlinf/workers/rollout/hf/huggingface_worker.py:147` vs
  `rlinf/workers/actor/embodied_fsdp_actor_worker.py:118`).
- **Device:** `model.to(...)` happens either inside the dispatcher (via
  `Worker.torch_device_type`, `rlinf/models/__init__.py:368-373`) or must be done by the
  adapter if it calls `dreamzero.get_model` directly. The patched VAE also needs CUDA at
  construction time (§1.4), so a CPU-only environment cannot even build the model
  **[V]**.

### Minimal plain-Python sequence (no Ray runtime, no Hydra job)

```python
import torch
from omegaconf import OmegaConf

# 1. architecture + checkpoint + runtime overrides
cfg = OmegaConf.load("RLinf/examples/embodiment/config/model/dreamzero_5b.yaml")
cfg = OmegaConf.merge(cfg, OmegaConf.create({
    "model_type": "dreamzero",
    "precision": "bf16",
    "is_lora": False,
    "model_path": "<checkpoint dir containing model.safetensors.index.json>",
    "metadata_json_path": "<checkpoint>/experiment_cfg/metadata.json",
    "tokenizer_path": "google/umt5-xxl",      # or a local umt5 path; never null
    "embodiment_tag": "libero_sim",
    "action_horizon": 16, "num_action_per_block": 16, "num_action_chunks": 16,
    "action_dim": 32, "max_action_dim": 32, "max_state_dim": 64, "state_horizon": 1,
    "target_video_height": 160, "target_video_width": 320,
}))

# 2. RLinf's own checkpoint-merge (folds in <model_path>/config.json, asserts required keys)
from rlinf.models.embodiment.dreamzero.dreamzero_config import validate_dreamzero_sft_model_cfg
cfg = validate_dreamzero_sft_model_cfg(cfg)

# 3. build. Either the dispatcher (adds .to(device), honours precision/is_lora):
from rlinf.models import get_model
policy = get_model(cfg)
# or, to control dtype/device yourself:
# from rlinf.models.embodiment.dreamzero import get_model as build
# policy = build(cfg, torch.bfloat16).to("cuda:0")

policy.eval()

# 4. one forward per observation
import numpy as np
actions, _ = policy.predict_action_batch(
    {"main_images": agentview[None], "wrist_images": wrist[None],
     "states": state[None].astype(np.float32), "task_descriptions": [instruction]},
    mode="eval",
)   # -> np.ndarray[1,16,7]
```

No Ray runner, placement, channel, worker or FSDP wrapper is needed. The only reason a
larger wrapper could be required is the §1.5 load-order problem, in which case the
adapter must inline the body of `dreamzero/__init__.py:get_model` (patcher, config,
metadata, transforms, construct, **promote**, load, `.to(dtype)`) instead of calling it.

---

## 4. Dependencies

### (a) What RLinf's installer actually pins for this path

`requirements/install.sh` for `--model dreamzero` with `--env libero`
(`install.sh:2314-2346`) does **[V]**:

1. `uv sync --extra embodied` — `RLinf/pyproject.toml`. Relevant: `torch>=2.5.0` (`:34`)
   overridden to `torch==2.11.0 / torchvision==0.26.0 / torchaudio==2.11.0`
   (`:170-174`); `ray[default]>=2.47.0` (`:33`); `hydra-core<1.4.0.dev8` (`:36`);
   `transformers<=4.57.6` (`:93`); `peft` unpinned (`:95`); `accelerate` (`:45`);
   `torchcodec>=0.11,<0.12` (`:174`).
2. `uv pip install -r requirements/embodied/models/dreamzero.txt` **[V]**:
   `numpy==1.26.4`, `tianshou==0.5.1`, `albumentations==1.4.18`, `ftfy==6.3.1`,
   `diffusers==0.37.1`, `peft==0.19.1`, `lerobot==0.3.3`, `torchcodec==0.2`,
   `decord==0.6.0`.
3. `python -m pip install -e "$dreamzero_path" --no-deps --ignore-requires-python`
   (`install.sh:2311`) — groot is installed **without** its own dependency resolution,
   from the pinned dreamzero commit `ab790c198fbce33503358efbbd4187ce9a89adf3`
   (`install.sh:2304-2307`). There is also a `maybe_build_decord_from_source`
   (`:1775-1808`) and `install_flash_attn` (`:2328-2334`).

### (b) What the `groot` / upstream dreamzero `pyproject.toml` pins

`dreamzero/pyproject.toml:21-100` **[V]**: `torch==2.8.0`, `torchvision==0.23.0`,
`torchaudio==2.8.0`, `transformers==4.51.3`, `peft==0.5.0`, `diffusers==0.30.2`,
`numpy==1.26.4`, `tianshou==0.5.1`, `albumentations==1.4.18`, `hydra-core`,
`ray[default]==2.47.1`, `einops==0.8.1`, `ftfy`, `datasets==3.6.0`, `decord2`,
`sentencepiece`, `tiktoken`, `opencv-python==4.8.0.74`, plus a `gpu` extra pinning a
**prebuilt flash-attn 2.8.3+cu12torch2.8** wheel and a `training` extra with `deepspeed`.
Because RLinf installs it with `--no-deps`, **none of these pins take effect in the RLinf
virtualenv** — they describe the upstream dreamzero environment, not the evaluated one.

### (c) Conflicts

| package | RLinf (effective for the eval venv) | dreamzero `pyproject.toml` | conflict |
|---|---|---|---|
| torch | `==2.11.0` (override) | `==2.8.0` | yes |
| torchvision | `==0.26.0` | `==0.23.0` | yes |
| transformers | `<=4.57.6` | `==4.51.3` | yes (compatible if the solver lands ≤4.51.3, but the pins are not the same) |
| peft | `==0.19.1` (dreamzero.txt, overriding the unpinned extra) | `==0.5.0` | yes |
| diffusers | `==0.37.1` | `==0.30.2` | yes |
| torchcodec | `>=0.11,<0.12` (pyproject) vs `==0.2` (dreamzero.txt) | — | **internal RLinf conflict** |
| numpy | `==1.26.4` | `==1.26.4` | no |
| tianshou | `==0.5.1` | `==0.5.1` | no |
| albumentations | `==1.4.18` | `==1.4.18` | no |
| ray | `>=2.47.0` | `==2.47.1` | no (bound compatible) |
| flash-attn | not in any requirements file; `install_flash_attn` picks `2.7.4.post1`, or `2.8.3` for torch≥2.8, wheel name derived from the *installed* torch (`install.sh:1410-1448,1489-1518`) | prebuilt 2.8.3+cu12torch2.8 (gpu extra) | the pinned dreamzero wheel matches torch 2.8, not 2.11 |

`RLinf/docs/action-eval/README.md:17-19` already records the Torch/TorchCodec/PEFT/
Diffusers/Transformers divergence **[V]**.

### Import-time requirements of the inference path specifically

Verified by reading the imports of every module on the path
(`dreamzero_policy.py:18-29`, `dreamzero/__init__.py:16-32`,
`data_transforms/__init__.py:15-37`, `dreamzero_cotrain.py:1-24`,
`wan_flow_matching_action_tf.py:1-50`, `wan_video_dit_action_casual_chunk.py:1-30`,
`wan2_1_attention.py:1-37`, `data/transform/video.py:1-5`,
`data/transform/state_action.py:1-11`, `data/transform/concat.py:1-8`) **[V]**:

`torch` (≥2.5 because of `torch.nn.attention.flex_attention`), `torchvision`
(`torchvision.transforms.v2`, `T.CenterCrop`, `T.Resize`), `numpy`, `einops`,
`transformers` (`AutoTokenizer`, `AutoProcessor`, `ProcessorMixin`, `PretrainedConfig`,
`PreTrainedModel`, `BatchFeature`, `DataCollatorMixin`), `accelerate`
(`load_checkpoint_and_dispatch`, imported unconditionally at
`wan_flow_matching_action_tf.py:7`), `peft` (`LoraConfig`, `get_peft_model` at `:11`,
**even though `is_lora` is false**), `diffusers` (`ConfigMixin`, `ModelMixin` in the DiT
module), `hydra-core` (`hydra.utils.instantiate`), `omegaconf`, `pydantic` (the transform
classes are pydantic models), `safetensors`, `huggingface_hub` (`hf_hub_download`),
`tianshou` (`tianshou.data.Batch` in `dreamzero_policy.py:21`), `ray` (via
`rlinf.utils.logging`, §3), `albumentations` + `opencv-python` (imported unconditionally
in `data/transform/video.py:3` even though the libero chain uses the torchvision
backend), `ftfy`, `regex`, `dm-tree` (`tree`), `tqdm` (RLinf VAE patch), `sentencepiece`
+ `tokenizers` (for the umt5 tokenizer), and `Pillow` (transitively through torchvision).

**Not required at import time nor at inference:**

- `flash-attn`: `wan2_1_attention.py:9-23` wraps all imports in `try/except
  ModuleNotFoundError`, the default backend is `"FA2"` (`:222-224`) and
  `flash_attention()` falls back to a pure-PyTorch SDPA implementation when
  `_gpu_supports_flash_attention()` is false (`:40-48,51-77,121-132`) **[V]**. So the
  path runs without flash-attn (the fallback respects `causal=True`); it is installed for
  speed and is the validated configuration. Note also that the SDPA fallback ignores
  `q_lens/k_lens` for padded/varlen calls, which do occur in some flash paths
  (`:62-66,336-346`), so "runs" is not the same as "numerically identical".
- `apex`: **no import anywhere in `groot`** (`grep` over `dreamzero/groot` found none)
  **[V]**; RLinf's dreamzero install path does not call `install_apex` either
  (`install.sh:2314-2346`) **[V]**.
- `natten`: same, no occurrence in `groot` **[V]**.
- `transformer_engine` / TensorRT: optional, guarded (`wan2_1_attention.py:27-33`,
  `ENABLE_TENSORRT` env in `wan_video_dit_action_casual_chunk.py:28`) **[V]**.
- `deepspeed`: training extra only (`groot/vla/experiment/*`) **[V]**.
- `lerobot`, `decord`, `torchcodec`, `av`: dataset/video-decoding only; the inference path
  receives uint8 arrays **[V]**.
- `pytorch3d`: imported inside `RotationTransform.__init__`
  (`state_action.py:44`), which is only constructed when `target_rotations` is non-empty
  (`:705-718`) — empty for LIBERO **[V]**.

### Can one isolated venv satisfy the inference path?

Yes in principle, and RLinf's `install.sh` already describes it, but not from the two
manifests as written: they conflict on torch/torchvision/peft/diffusers/transformers and
RLinf is internally inconsistent on torchcodec. A single venv is achievable because
groot is installed with `--no-deps`, so the dreamzero pins are advisory. Practical
requirements: Python 3.10/3.11, torch ≥2.5 with CUDA, torchvision matching torch, plus
the import list above. The unresolved question is whether groot (the package pinned to
dreamzero commit `ab790c198fbce33503358efbbd4187ce9a89adf3`, `install.sh:2304-2307`)
actually *runs* on torch 2.11 — e.g.
`torch.nn.attention.flex_attention` and Dynamo behaviour changed across 2.5→2.11
**[U]**. A defensible experiment should either adopt RLinf's stack verbatim (torch
2.11 + peft 0.19.1 + diffusers 0.37.1 + transformers ≤4.57.6) or pin something
explicitly and record it, not leave the resolver to choose.

---

## 5. Weights and assets

| asset | needed at inference? | how configured |
|---|---|---|
| Wan2.2 VAE (`WanVideoVAE38`, z_dim 48, dim 160) | **architecture yes; external weights no** when the full checkpoint is present. `skip_component_loading=True` makes `WANPolicyHead.__init__` skip `ensure_file(...)`/`load_state_dict` for T5, CLIP, VAE and the DiT init (`wan_flow_matching_action_tf.py:242-324`) — but the modules are still built by `instantiate` (`:174-176,237`) and their weights come from the shards **[V]**. The patched VAE also allocates `mean`/`std` on `torch.device("cuda")` (`patch/wan_video_vae.py:44-48`) **[V]**. `vae_cfg` at `dreamzero_5b.yaml:105-110`. |
| umt5-xxl text encoder (`models_t5_umt5-xxl-enc-bf16.pth`) | **no** (weights from shards, `skip_component_loading=True`); architecture `WanTextEncoder` is instantiated from `text_encoder_cfg` (`dreamzero_5b.yaml:88-91`) **[V]**. It *is* required if the checkpoint has no full weights, or if one calls `post_initialize`/component loading. |
| CLIP image encoder (`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`) | **no** for the same reason; architecture from `image_encoder_cfg` (`dreamzero_5b.yaml:66-69`) **[V]**. Used at inference (`encode_image` → `self.image_encoder.encode_image`, `wan_flow_matching_action_tf.py:578`) with weights from the checkpoint. |
| action-DiT init (`Wan-AI/Wan2.2-TI2V-5B`, `diffusion_pytorch_model*.safetensors`) | **no** when `skip_component_loading=True`; architecture from `diffusion_model_cfg` (`dreamzero_5b.yaml:43-61`) **[V]** |
| **umt5 tokenizer** | **YES, always** — `HuggingfaceTokenizer` is constructed lazily but used by `DreamZeroCollator.collate_batch` on every prediction (`dream_transform.py:76-84`, `dataloader.py:71`) **[V]**. Configure with `tokenizer_path`; must be non-null. The checkpoint `config.json` supplies an absolute training-machine path — prefer an explicit local path or `google/umt5-xxl` (needs HF access or a warm cache; `HuggingfaceTokenizer` passes `local_files_only=True` only when `name` is an existing directory, `dreamzero_cotrain.py:46-52`) **[V]**. |
| **`metadata.json`** (normalisation statistics, video resolutions, modality shapes) | **YES, always** — `load_dreamzero_dataset_metadata` (`data_transforms/__init__.py:153-191`), consumed by `data_transforms.set_metadata` before any prediction. Path: `metadata_json_path` if set, else `<model_path>/experiment_cfg/metadata.json` **[V]**. The HF checkpoint ships it at `experiment_cfg/metadata.json` **[V]** (confirmed by the task description and by `dreamzero_save_helper`, `utils/ckpt_convertor/fsdp_convertor/utils.py:88-115`). |

No copy of `metadata.json` exists locally, so its exact contents are **not verified**.
From the generator and the consuming code, the expected structure is **[V]**/*[I]*:

```jsonc
{
  "libero_sim": {
    "statistics": {
      "state":  { "state":   {"min":[...8],"max":[...8],"mean":[...8],"std":[...8],"q01":[...8],"q99":[...8]} },
      "action": { "actions": {"min":[...7],"max":[...7],"mean":[...7],"std":[...7],"q01":[...7],"q99":[...7]} }
    },
    "modalities": {
      "video": { "image": {"resolution":[H,W],"channels":3,"fps":f},
                 "wrist_image": {"resolution":[H,W],"channels":3,"fps":f} },
      "state":  { "state":   {"absolute":true,"rotation_type":null,"shape":[8],"continuous":true} },
      "action": { "actions": {"absolute":true,"rotation_type":null,"shape":[7],"continuous":true} }
    },
    "embodiment_tag": "libero_sim"
  }
}
```

(`toolkits/lerobot/generate_dreamzero_metadata.py:154-165` for the six statistics keys,
`:376-418` for the metadata shape, `:292-300` for `resolution` from `meta/info.json`;
`groot/vla/data/schema/lerobot.py:217-270` for the required fields; `video.py:250-256,
276-288` for the hard requirement that the incoming frame size equal
`modalities.video.<key>.resolution`, and `concat.py:250-258` /
`state_action.py:673-703` for the shape/statistics consumers) **[V]**.

Consequences to check on the real file:

- `q01`/`q99` must be present for `state.state` and `action.actions`; they are declared
  required (`lerobot.py:217-223`) so their absence fails loudly at
  `set_metadata` **[V]**.
- `modalities.video.image.resolution` and `...wrist_image.resolution` must equal the
  benchmark's rendered camera size. The action-eval release protocol uses
  `resolution: [256, 256]` (`action-eval/configs/protocols/dreamwam-release-v1.yaml`)
  and creates the LIBERO env at that size (`backend.py:317-330`) **[V]**; RLinf's LIBERO
  env defaults to 256×256 (`rlinf/envs/sim/libero/libero_env.py:866`) **[V]**. A metadata
  resolution of e.g. `[224,224]` would abort inside `VideoCrop.check_input` **[V]**, a
  loud failure, which is the good case.
- `action.actions.shape` must be `[7]` and `state.state.shape` `[8]`; `ConcatTransform`
  asserts the incoming width against `action_dims`/`state_dims` (`concat.py:132-164`)
  **[V]**. A wrong width fails loudly.

---

## 6. The `dreamzero`-only alternative (groot alone, no RLinf)

**Partially feasible, and the gap is larger than "just the transforms".**
`DreamZeroPolicy` does subclass `groot.vla.model.dreamzero.base_vla.VLA` **[V]**, and
groot's own `VLA.from_pretrained` (`base_vla.py:511-589`) is written to load exactly this
kind of artifact **[V]**:

```python
config = VLAConfig(**config_dict)
_mark_full_checkpoint_component_loading(config.action_head_cfg)   # sets skip_component_loading=True
...
model = cls(config)
...
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
```

What is missing if RLinf is not installed:

1. **The scalar-parameter promotion** — `_promote_scalar_params_to_1d` lives in RLinf
   (`RLinf/rlinf/models/embodiment/dreamzero/__init__.py:57-73`) and must run **before**
   the same `load_state_dict` that groot calls internally, so `VLA.from_pretrained` cannot
   be used unmodified (§1.5, **[I]**). Reimplementing it is ~10 lines.
2. **The rollout observation mapping** — `convert_rollout_env_obs` /
   `RolloutObsLayout` / `convert_rollout_env_obs_with_layout`
   (`RLinf/.../data_transforms/base.py:26-186`) must be reimplemented: map
   `main_images → video.image`, `wrist_images → video.wrist_image`,
   `states → state.state` as `[:,None,:]`, `task_descriptions → annotation.task`. ~60
   lines, mechanical.
3. **The horizontal multi-view layout** — groot's base `DreamTransform._prepare_video`
   builds a **2×2 grid** for multi-view (`dreamzero_cotrain.py:357-379`) **[V]**, which
   is not the LIBERO layout. RLinf overrides it
   (`data_transforms/dream_transform.py:213-221` → `libero_sim.py:72-83`) **[V]**. A
   subclass or monkey-patch is mandatory. ~15 lines.
4. **The prompt wrapper and the collator** — groot's own `collate`
   (`dreamzero_cotrain.py:92-164`) indexes `embodiment_tag_mapping[EmbodimentTag.AGIBOT.value]`
   and friends with no `libero_sim` branch, and its only fallthrough is
   `raise ValueError(f"Embodiment ID {...} not supported.")` at `:128,150` **[V]**. It
   also does not lowercase the instruction. RLinf's `DreamZeroCollator.collate_batch` +
   `format_training_prompt` + `normalize_instruction_text` replace it
   (`dataloader.py:54-88`, `libero_sim.py:67-70`, `data_transforms/__init__.py:81-91`)
   **[V]**. Reimplementing the template string by hand is the single highest-risk part of
   this alternative: a one-token difference in the prompt is a silent failure mode.
5. **`LiberoSimDataTransform` itself** — the modality config, the composed chain (including
   the 0.95 centre crop, the 256×256 resize, the q99 transforms, the concat order and the
   `binarize_gripper` flag) exist only in RLinf (`libero_sim.py:36-212`) **[V]**. groot
   provides the individual transform classes, so this is a re-composition, ~80 lines, but
   every value must be reproduced exactly.
6. **The metadata loader** — RLinf's version deliberately uses
   `DatasetMetadata.model_construct` to dodge a pydantic enum-binding problem caused by
   its own `EmbodimentTag` patch (`data_transforms/__init__.py:182-191`) **[V]**. Using
   groot's enum, plain `DatasetMetadata.model_validate(blob["libero_sim"])` should work,
   because groot's `EmbodimentTag` already contains `LIBERO_SIM = "libero_sim"`
   (`groot/vla/data/schema/embodiment_tags.py:185`) **[V]** and `DatasetMetadata` has no
   restricting validator (`lerobot.py:262-270`) **[V]**. Low risk.
7. **`unapply` + gripper binarisation + `_actions_from_unapply`** — RLinf's
   `dreamzero_policy.py:120-259` **[V]**. ~40 lines. This is where a mistake silently
   changes action scale (§7).
8. **`predict_action_batch` itself** — the env-obs → Batch → transform → forward →
   unapply sequence (`dreamzero_policy.py:261-305`) **[V]**. ~25 lines.

So: the groot-only adapter is **not blocked**, it is a re-implementation of roughly
250 lines of RLinf code that is mostly *configuration*, not algorithm. The critical
asymmetry is that RLinf's copies are the *validated* ones; a hand-copy silently diverges
wherever a constant is wrong (prompt text, 0.95 crop, 256 resize, 160×320 target, q99
inverse, gripper binarisation order). The recommendation is therefore: use RLinf's code
as the primary path, and if RLinf cannot be installed in the evaluation venv, treat the
groot-only port as a parity project with an explicit numerical-parity test against RLinf,
not as a quick fallback.

One more subtlety for the groot-only path: `VLA` is registered with
`AutoConfig.register("vla", VLAConfig)` / `AutoModel.register(...)`
(`base_vla.py:626-628`) **[V]**, and `VLAConfig` ignores the `model_type: "dreamzero"`
in `config.json` (it becomes an attribute) **[V]**, so `from_pretrained` does not need
RLinf's config class.

---

## 7. Action semantics warning

**What the output is** **[V]**:

- `action_pred` from the head is a flow-matching sample of shape `[B,16,32]` in the q99
  normalised space, i.e. roughly `[-1,1]` per real dim and identically-`0`-clamped for the
  25 padded dims (`max_action_dim = 32`, `action.actions` is 7 wide —
  `dreamzero_cotrain.py:459-465` pads the *state*, and `ConcatTransform.unapply` slices the
  *action* at `concat.py:179-184`) **[V]**.
- After `unapply`, the 7 real columns are the **q99-denormalised training-dataset actions**
  — for LIBERO that is the raw `OSC_POSE` vector `(dx,dy,dz,dax,day,daz,gripper)` that
  LIBERO's controller consumes (action-eval's `ACTION_DIM = 7`, `backend.py:37-41`) **[V]**.
  The values are already in simulator units; no extra scaling exists anywhere in the
  RLinf path **[V]**.
- LIBERO's controller applies its own `output_max` scaling internally, so the numbers
  handed to `env.step` are exactly the numbers the demonstration dataset stored. The
  adapter must return them **as is** and must not rescale.
- `relative_action: false` in the checkpoint `config.json` and
  `dreamzero/__init__.py:166-169` parsing a missing `relative_action_keys` as `[]` mean the
  relative→absolute block is skipped **[V]**. The output is **not** relative to the input
  state and must not have the state added — unless the checkpoint's `config.json` were to
  say `relative_action: true`, which the task statement says it does not.
- The gripper column is q99-inverted and then thresholded to `±1` (§2.6) **[V]**.

### Every place a wrong assumption fails silently

1. **Skipping the q99 inverse.** The LIBERO action distribution is roughly within
   `[-1,1]`, so the q99 map is close to identity and the *normalised* values look like
   plausible small OSC deltas. Feeding normalised actions to the simulator produces
   slightly mis-scaled (and for dims where `q99-q01` differs from 2, mis-scaled by
   `1/((q99-q01)/2)`) motion — a low success rate with no exception anywhere. The only
   check is to compare the returned action range against the metadata
   `statistics.action.actions.{q01,q99}`.
2. **Normalising instead of denormalising** (or applying `unapply` twice): same class of
   silent error.
3. **Adding the state** (`obs["state"] + action`) when `relative_action` is false: the
   first 6 dims become eef-pose-scale values, the arm jumps or freezes. `unapply` only
   does this when the config says so **[V]**, so an adapter must not "help".
4. **Swapping `main_images`/`wrist_images`.** No error; the prompt claims exterior-left /
   wrist-right while the pixels are reversed. Expected: near-zero success.
5. **Flipping the images again.** The benchmark already applies `image[::-1, ::-1]`
   (`backend.py:106`), matching RLinf's LIBERO env (`utils.py:90,108`) **[V]**. A second
   flip feeds an upside-down view.
6. **Wrong `target_video_height/width`.** 176×320 (the `dreamzero_5b.yaml` default) vs
   160×320 (the checkpoint) changes the latent grid and therefore the DiT positions; the
   model still runs and returns actions **[V]**. The fallback path at
   `wan_flow_matching_action_tf.py:1027-1031` silently picks 176×320 when the config field
   is unset and `frame_seqlen ∈ (50,55)`, so an adapter that forgets to set these keys is
   silently wrong.
7. **Writing your own prompt.** A missing/duplicated prefix, suffix, or the missing
   lowercase produces a valid token sequence and a degraded policy
   (`libero_sim.py:67-70`, `data_transforms/__init__.py:81-91`) **[V]**.
8. **Wrong action-horizon / chunk semantics.** The checkpoint emits 16 actions. The
   action-eval protocol `dreamwam-release-v1` executes only the first `replan_steps = 10`
   (`runner/episode.py:189-199`) **[V]**, while RLinf's own eval consumed
   `num_action_chunks = 16`. This is a *protocol* decision to be declared in the
   experiment config, not a silent bug — but reporting "the RLinf number" while executing
   10 of 16 is a mislabelled comparison.
9. **Declaring the wrong `action_horizon` in `describe()`.** The SDK checks only
   `action_horizon >= replan_steps` (`adapter.py:103-107`) **[V]**, so an optimistic 16
   with an actual 8 would pass the handshake and then abort mid-episode with a
   `policy_contract` error at `episode.py:189`.
10. **Gripper convention.** The benchmark's warm-up holds the gripper open with
    `action[-1] = -1.0` (`backend.py:193-197`) **[V]**, i.e. `-1` = open; RLinf binarises
    `>0 → +1` (closed) **[V]**. Inverting the sign is silent.
11. **State layout.** The 8-dim vector is `[eef_pos(3), axis_angle(3), gripper_qpos(2)]`
    on both sides (`backend.py:66-84`; `rlinf/envs/sim/libero/libero_env.py:752-760`;
    identical `quat2axisangle` formula in `backend.py:50-63` and
    `rlinf/envs/sim/libero/utils.py:112-136`, both robosuite's x,y,z,w convention) **[V]**.
    Reordering it (e.g. quaternion instead of axis-angle) is silent: the q99 statistics
    are indexed positionally.

---

## Minimal adapter skeleton

Place at `<model repo>/evaluation/action_eval/infer.py`; the action-eval worker imports it
by file path (`action-eval/packages/policy-sdk/src/action_eval_sdk/worker.py:106-133`)
**[V]**, so nothing has to be installed as a package.

The `config` argument is the merged option dict `{"model_root", "model_config",
"checkpoint", **policy.options}` (`action-eval/src/action_eval/runner/run.py:548-580`)
**[V]**. `model_root` is the *fork* root, not the checkpoint — take the checkpoint from
`config["checkpoint"]`.

I recommend the adapter live in the **RLinf fork** and be registered with
`policy.repo_root` pointing there, so `model_root` gives direct access to
`examples/embodiment/config/model/dreamzero_5b.yaml`. Placing it in the `dreamzero` fork
also works, as long as the venv that runs it has RLinf installed and the architecture YAML
is reachable through `model_config` or an option.

```python
"""action-eval adapter for RLinf DreamZero WAN2.2-5B LIBERO-SFT.

Design rule: this file contains NO modelling code. Preprocessing (image flip, resize,
crop, multi-view concat, prompt template, q99 normalisation) and postprocessing
(denormalisation, gripper binarisation) are done by RLinf's DreamZeroPolicy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from action_eval_sdk import (
    ACTION_PROFILE_LIBERO_OSC_POSE_V1,
    OBSERVATION_PROFILE_LIBERO_2CAM_V1,
    RNG_MODE_FIXED_PER_PREDICT,
    PolicyDescription,
    Prediction,
)

# Verified contract: images are uint8 [H,W,3] already flipped by the benchmark.
# state is float32 [8] = eef_pos(3), axis_angle(3), gripper_qpos(2).
# RLinf's env produces exactly the same layout and flip.

_ARCH_YAML_REL = "examples/embodiment/config/model/dreamzero_5b.yaml"


def create_policy(config: dict, device: str):
    return DreamZeroAdapter(config, device)


class DreamZeroAdapter:
    MODEL_YAML_ACTION_HORIZON = 16  # value in dreamzero_5b.yaml; overridden by the ckpt

    def __init__(self, config: dict, device: str):
        self._config = dict(config)
        self._device = device
        ckpt = Path(config["checkpoint"]).expanduser().resolve()
        self._ckpt = ckpt

        ckpt_cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
        repo_root = Path(config["model_root"]).expanduser().resolve()
        arch_path = config.get("arch_yaml") or (repo_root / _ARCH_YAML_REL)

        cfg = OmegaConf.load(str(arch_path))
        cfg = OmegaConf.merge(cfg, OmegaConf.create({
            "model_type": "dreamzero",
            "precision": config.get("precision", "bf16"),
            "is_lora": False,
            "model_path": str(ckpt),
            "embodiment_tag": config.get("embodiment_tag", "libero_sim"),
            # UNCERTAIN: set explicitly; a null here loses to the checkpoint's
            # absolute training-machine path (see INFERENCE-PATH.md 1.3).
            "metadata_json_path": str(ckpt / "experiment_cfg" / "metadata.json"),
            # UNCERTAIN: prefer a local umt5 dir; "google/umt5-xxl" needs HF access.
            "tokenizer_path": config.get("tokenizer_path")
                               or self._usable_tokenizer_path(ckpt_cfg)
                               or "google/umt5-xxl",
            # Architecture knobs mirrored from the checkpoint config.json where present.
            "action_horizon": int(ckpt_cfg.get("action_horizon", self.MODEL_YAML_ACTION_HORIZON)),
            "num_action_per_block": int(ckpt_cfg.get("num_action_per_block", 16)),
            "num_action_chunks": int(ckpt_cfg.get("num_action_chunks", 16)),
            "action_dim": int(ckpt_cfg.get("max_action_dim", 32)),
            "max_action_dim": int(ckpt_cfg.get("max_action_dim", 32)),
            "max_state_dim": int(ckpt_cfg.get("max_state_dim", 64)),
            "state_horizon": int(ckpt_cfg.get("state_horizon", 1)),
            # 160/320, NOT the model yaml default 176/320 (see INFERENCE-PATH.md 7.6)
            "target_video_height": int(ckpt_cfg.get("target_video_height", 160)),
            "target_video_width": int(ckpt_cfg.get("target_video_width", 320)),
            "relative_action": bool(ckpt_cfg.get("relative_action", False)),
        }))

        from rlinf.models.embodiment.dreamzero.dreamzero_config import (
            validate_dreamzero_sft_model_cfg,
        )
        cfg = validate_dreamzero_sft_model_cfg(cfg)     # folds in <model_path>/config.json
        self._action_horizon = int(cfg.action_horizon)  # declare what the model returns

        # Option A (try first): RLinf's loader.
        from rlinf.models import get_model
        self._policy = get_model(cfg)                   # precision/is_lora/.to(device)
        self._policy.eval()

        # Option B (if Option A raises a size mismatch for promoted scalars):
        #   inline the body of rlinf.models.embodiment.dreamzero.get_model and flip the
        #   order of its last three statements, keeping model_path set so that
        #   skip_component_loading stays True (setting model_path=None would instead
        #   download the generic WAN components and load no SFT weights at all):
        #     from rlinf.models.embodiment.dreamzero import (
        #         _configure_torch_dynamo_for_dreamzero_inference,
        #         _promote_scalar_params_to_1d, DreamZeroConfig, DreamZeroPolicy,
        #         build_dreamzero_composed_transform, load_dreamzero_dataset_metadata,
        #     )
        #     from rlinf.utils.patcher import Patcher
        #     ... Patcher.add_patch(...) exactly as get_model does ...; Patcher.apply()
        #     dc = DreamZeroConfig(**OmegaConf.to_container(cfg, resolve=True))
        #     dc.action_head_cfg["config"]["skip_component_loading"] = True   # full ckpt
        #     dc.action_head_cfg["config"]["defer_lora_injection"] = False
        #     md = load_dreamzero_dataset_metadata(cfg)
        #     dt = build_dreamzero_composed_transform(cfg, cfg.tokenizer_path)
        #     dt.set_metadata(md); dt.eval()
        #     dc.data_transforms = dt; dc.embodiment_tag = str(cfg.embodiment_tag)
        #     model = DreamZeroPolicy(config=dc)
        #     _promote_scalar_params_to_1d(model)                 # <-- BEFORE the load
        #     model.load_state_dict(<merged shards>, strict=False)
        #     model = model.to(self._device, dtype=torch.bfloat16)

        # Warm up once: RLinf wraps four DiT attention helpers in torch.compile(
        # mode="reduce-overhead"), so the first real predict pays compilation. Better to
        # pay it here than against first_predict_timeout_s.
        self._warmup()

    # ------------------------------------------------------------------ lifecycle
    def describe(self) -> PolicyDescription:
        return PolicyDescription(
            observation_profile=OBSERVATION_PROFILE_LIBERO_2CAM_V1,
            action_profile=ACTION_PROFILE_LIBERO_OSC_POSE_V1,
            rng_mode=RNG_MODE_FIXED_PER_PREDICT,   # action head seeds 1140 every call
            action_horizon=self._action_horizon,
            proprio_dim=8,
            action_dim=7,
            camera_keys=("agentview", "wrist"),
            fingerprint={
                "checkpoint": str(self._ckpt),
                "config_json": str(self._ckpt / "config.json"),
                "policy_class": type(self._policy).__name__,
                "precision": "bf16",
            },
            notes="RLinf DreamZeroPolicy; images are benchmark-flipped; actions are "
                  "q99-denormalised LIBERO OSC deltas with binarised gripper.",
        )

    def reset(self, episode: dict) -> None:
        # predict is stateless for single-frame observations (the T==1 branch resets
        # current_start_frame and rebuilds the caches every call), but clear the cached
        # state explicitly so an episode never inherits it.
        ah = getattr(self._policy, "action_head", None)
        if ah is not None:
            ah.language = None
            ah.current_start_frame = 0
            ah.clip_feas = None
            ah.ys = None
            ah.kv_cache1 = ah.kv_cache_neg = None
            ah.crossattn_cache = ah.crossattn_cache_neg = None

    def predict(self, observation: dict) -> Prediction:
        images = observation["images"]
        state = np.asarray(observation["state"], dtype=np.float32).reshape(1, 8)
        instruction = str(observation["instruction"])
        env_obs = {
            "main_images": np.ascontiguousarray(images["agentview"])[None],   # exterior LEFT
            "wrist_images": np.ascontiguousarray(images["wrist"])[None],      # wrist RIGHT
            "states": state,
            "task_descriptions": [instruction],       # bare text; RLinf wraps+lowercases it
        }
        t0 = time.perf_counter()
        actions, _ = self._policy.predict_action_batch(env_obs, mode="eval")
        dt = time.perf_counter() - t0
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise RuntimeError(f"unexpected action shape {actions.shape}")
        chunk = actions[0]                            # [16,7], simulator units
        if not np.isfinite(chunk).all():
            raise RuntimeError("non-finite action")
        return Prediction(
            actions=chunk,
            timing={"predict_seconds": dt},
            diagnostics={"action_min": float(chunk.min()), "action_max": float(chunk.max()),
                         "gripper_first": float(chunk[0, -1])},
        )

    def close(self) -> None:
        del self._policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _usable_tokenizer_path(ckpt_cfg: dict) -> str | None:
        p = ckpt_cfg.get("tokenizer_path")
        return p if p and Path(str(p)).exists() else None

    def _warmup(self) -> None:
        zeros = np.zeros((1, 256, 256, 3), dtype=np.uint8)
        try:
            self._policy.predict_action_batch(
                {"main_images": zeros, "wrist_images": zeros,
                 "states": np.zeros((1, 8), dtype=np.float32),
                 "task_descriptions": ["warm up"]},
                mode="eval",
            )
        except Exception:
            # A failed warm-up must not be mistaken for a load failure; the first real
            # predict will surface the same error as a policy_adapter error.
            pass
```

Uncertainties deliberately left visible in the skeleton:

- **The metadata/experiment YAML knobs that must come from the checkpoint**
  (`action_horizon`, `num_action_per_block`, `target_video_height/width`) are read from
  `config.json` with the LIBERO values as defaults. That they equal the training values is
  **[I]**, not verified locally.
- **`metadata_json_path` and `tokenizer_path` handling** is a workaround for RLinf's
  merge precedence; confirm it on the real checkpoint (§1.3).
- **Option A vs Option B** depends on the promoted-scalar question (§1.5) **[U]**.
- **`arch_yaml`**: if the adapter is not run from an RLinf checkout, the architecture YAML
  must be supplied (via `model_config` or an options key) or baked into the adapter as a
  literal dict.
- **`reset()` touches private attributes** (`action_head.language`, `kv_cache1`, …). They
  are the fields the action head itself uses (`wan_flow_matching_action_tf.py:185-201`,
  `1043-1056`), so this is stable for this commit but is not a public API.
- **`describe().action_horizon`** is the true chunk length (16). Whether the experiment
  executes 10 (protocol) or 16 (RLinf eval) is a protocol/experiment choice, not an
  adapter choice.

---

## Open questions that only a GPU run (or the downloaded checkpoint) can settle

1. **Does the HF repo contain `model.safetensors.index.json`?** If not, RLinf silently
   falls back to the generic component-initialised DreamZero and only logs a warning
   (`__init__.py:135-201`) — the run would "work" with near-zero success.
2. **Do the shards contain FSDP-promoted scalar parameters shaped `[1]`** (e.g.
   `action_head.image_encoder.model.log_scale`)? If yes, RLinf's own
   `get_model(model_path=...)` raises a size mismatch because
   `_promote_scalar_params_to_1d` runs after loading (§1.5) **[I]**.
3. **Exact contract of `metadata.json`**: presence of `q01`/`q99` for
   `state.state` and `action.actions`, the recorded video `resolution` (must be the
   256×256 the benchmark renders), and `action.actions.shape == [7]`.
4. **Does the checkpoint `config.json` pin `target_video_height: 160` inside
   `action_head_cfg.config`**, or only at top level? Only the nested value reaches
   `WANPolicyHeadConfig` (`wan_flow_matching_action_tf.py:1025`), and the fallback is
   176×320 (§7.6).
5. **Is the tokenizer path in `config.json` reachable on the evaluation server**, or must
   `google/umt5-xxl` be resolved (network/cache)?
6. **First-predict wall time.** Four `torch.compile(mode="reduce-overhead")` wrappers plus
   CUDA-graph capture can take a long time; action-eval allows 120 s per predict and
   3×120 s for the first (`packages/policy-sdk/src/action_eval_sdk/client.py:79-85`)
   while model load allows 600 s (`startup_timeout_s`). The LLM-level advice is to keep
   the warm-up but also raise `first_predict_timeout_s`/`startup_timeout_s` in the
   experiment YAML if measurement shows they are tight.
7. **VRAM and dtype.** The whole model is cast to bf16 by the dispatcher
   (`models/__init__.py:365-373`), while the checkpoint's own `config.json` says
   `precision: fp32`; the DiT cannot run fp32 weights with bf16 activations
   (`lazy_joint_video_action` casts video/state/action to bf16 at
   `wan_flow_matching_action_tf.py:1012,1019,1022`). Confirm that bf16 is what the
   released number used, and that the whole stack fits on the target GPU. Dtype choice is
   a protocol-level provenance field, not a tuning knob.
8. **Numerical parity of the SDPA fallback** if `flash-attn` is not installed.
9. **Whether the model was trained with `replan_steps = 10` or 16** for the reporting
   protocol; the action-eval release protocol fixes 10, RLinf's own eval used 16
   (`num_action_chunks`), so the two are not the same experiment.
10. **Item 7.1's magnitude.** Sample the q99 statistics of the real
    `action.actions` to know how far a forgotten denormalisation would move the actions —
    it determines how silent that bug is.

No statement in this document should be read as a measured result. Everything above is a
static reading of the three local repositories.
