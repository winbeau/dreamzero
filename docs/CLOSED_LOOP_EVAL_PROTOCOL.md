# Closed-loop DROID simulation protocol

Status: infrastructure prepared on 2026-08-30; rollout execution requires an
RTX-capable simulation node.

## Reproducible simulator environment

The official `arhanjain/sim-evals` source was fixed at commit `3a6b0e8` and
installed as a separate uv project under:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse/sim-evals
```

The environment uses Python 3.11.15, Isaac Lab 2.2.0, Isaac Sim 5.0.0.0, and
the locked 237-package resolution from upstream.  It occupies about 17 GiB.
The official `owhan/DROID-sim-environments` assets occupy about 88 MiB under
`sim-evals/assets`.

Upstream `flatdict==4.0.1` imports `pkg_resources` without declaring the build
dependency.  The external simulator `pyproject.toml` therefore needs:

```toml
[tool.uv.extra-build-dependencies]
flatdict = ["setuptools<81"]
```

Use the shared cache and the interactive shell definition of the proxy:

```bash
bash -ic 'proxyon && \
  UV_CACHE_DIR=/data/chenjiayu/.cache/uv \
  /data/chenjiayu/.local/bin/uv sync --locked'

bash -ic 'proxyon && \
  UV_CACHE_DIR=/data/chenjiayu/.cache/uv \
  /data/chenjiayu/.local/bin/uv run hf download \
  owhan/DROID-sim-environments --repo-type dataset --local-dir assets'
```

## GPU requirement

The current H200 node cannot execute this RGB-camera simulator.  Isaac Sim's
official requirements state that GPUs without RT Cores, including A100 and
H100, are unsupported.  H200 is also a Hopper compute GPU without RT Cores.
Both local smoke attempts reached the RTX/Vulkan startup and failed with
`VK_ERROR_DEVICE_LOST`, while CUDA driver symbol checks passed.  Run the
simulator on an RTX/Ada/Blackwell visualization GPU instead.

Official requirement:
https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html

DreamZero inference may remain on the H200 node if the RTX simulator can reach
its WebSocket port.

## Evaluation runner fixes

`eval_utils/run_sim_eval.py` now:

- resets the simulator and policy history for every episode;
- accepts explicit `--device`, `--seed`, and `--max-steps` controls;
- moves action tensors onto the simulator device;
- optionally stops after automatic success;
- writes every episode video plus a crash-resilient `metrics.json` file.

The upstream simulator defines only a timeout termination and no reward or
success predicate.  `eval_utils/sim_eval_metrics.py` therefore records a
conservative center-in-container proxy for the three official scenes:

| Scene | Source | Target | XY threshold |
| --- | --- | --- | ---: |
| 1 | `rubiks_cube` | `_24_bowl` | 0.080 m |
| 2 | `_10_potted_meat_can` | `_25_mug` | 0.055 m |
| 3 | `_11_banana` | `small_KLT_visual_collision` | 0.160 m |

The JSON metric is a screening signal.  Saved videos remain the authoritative
success audit until the thresholds are validated on an RTX node.

## Matched sparse-versus-dense protocol

Use identical scene/seed pairs for the dense baseline and sparse candidate.
The current systems candidate is:

```text
historical keep ratio: 0.20
current visual keep ratio: 0.50
dense prefix/suffix: 5 / 5 layers
propagation radius/every: 1 / 5 sparse layers
recent dense frames: 2
```

Run all three scenes with the same seed range.  A minimal simulator command on
an RTX node is:

```bash
cd /path/to/DreamZero
/path/to/sim-evals/.venv/bin/python -m eval_utils.run_sim_eval \
  --episodes 20 \
  --scene 1 \
  --host H200_POLICY_HOST \
  --port 6000 \
  --device cuda:0 \
  --seed 0
```

Repeat for scenes 2 and 3, then repeat the identical seeds for the other
attention mode.  Report task success, paired success delta, wall-clock policy
latency, and end-to-end rollout time.  Keep full-budget sparse parity as a
separate exactness control.
