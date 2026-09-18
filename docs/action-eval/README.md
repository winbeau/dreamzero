# DreamZero LIBERO integration plan

This page records the future single-H200 evaluation of **RLinf DreamZero WAN2.2-5B LIBERO-SFT**. Execution is paused at the user's request; no new model weights, environment, adapter or GPU run has been validated. main remains the maintained branch, preserving existing DROID/sparse work.

## Sources and model identity

- RLinf source audited: `717dfb2cb9e59927fb9f5ea207ab63f43b119f1d`.
- Its DreamZero dependency reference: `ab790c198fbce33503358efbbd4187ce9a89adf3`. This is an upstream audit reference, not permission to reset main.
- Native HuggingFace path: RLinf `evaluations/libero/libero_spatial_dreamzero_eval.yaml` → DreamZeroPolicy.predict_action_batch → lazy_joint_video_action_causal. Prefer reviewing this path before adding SGLang.
- Candidate release: [RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000](https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000), reported HF revision `7416908d4e12c7a9fb3b7eb4fde7408fd2559e3c`. Revalidate before downloading. This is a repository revision, NOT a weight SHA-256.
- Step18000 documentation links redirect to Step26000; do not transfer old published scores to current weights. The public recipe points at four-suite data, but actual checkpoint training coverage lacks direct release evidence.

## Future stages

1. Review existing manifest/lock against RLinf dependencies without modifying the running 14B environment. Preserve exact sources and hashes; explain unavoidable changes.
2. Resolve Torch/TorchCodec, PEFT, Diffusers and Transformers compatibility. Current RLinf installer is not a complete reusable lock and may reset shared ~/.libero; do not run it blindly.
3. Pin/download the selected full checkpoint and matching metadata/tokenizer only after explicit resumed authorization; hash all assets. Override training-machine paths through config.
4. Implement the LIBERO observation/action adapter around the native policy. DROID mappings are not LIBERO mappings. Verify reset/KV-cache isolation and native parity.
5. Run one environment on physical GPU5 (logical cuda:0), then four-suite admission/timing. Published single-card configuration is not our hardware validation.
6. After budget approval, target 4 suites × 10 tasks × 50 initial states = 2,000 episodes. Retain every attempt; missing results never render as zero.

## Handoff and records

See [the authoritative two-model plan](https://github.com/winbeau/action-eval/blob/main/docs/plan/00-current-handoff.md). Future records must contain PLANNED/VERIFIED/FAILED/BLOCKED status, timestamp/timezone, exact code/lock/checkpoint identities, command, exit code, artifact location, observations and limitations. No command described here was executed as a DreamZero evaluation during this preparation.
