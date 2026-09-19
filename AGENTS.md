# DreamZero evaluation fork rules

This fork uses **main** as its maintained evaluation branch and GitHub default. Final user scope: **plan only, execution paused**. The future target is RLinf DreamZero WAN2.2-5B LIBERO-SFT, not the existing DROID 14B model. Preserve existing sparse experiment history; do not reset main to an upstream commit.

## Mandatory workflow

All projects: local clone/edit → local git add/commit/push on main → compute server git pull --ff-only → uv sync --locked → explicitly authorized training/evaluation. Never install or run models locally, modify application code on the server, or deploy source by copying around Git. Check the server tree before pulling; never discard edits. Keep data/weights/cache out of Git.

Reuse validated pyproject.toml and uv.lock byte-for-byte wherever compatible; pin new dependencies/build tools/source commits exactly, document necessary differences, never upgrade implicitly. Missing locks may be generated only in isolated server staging, returned for local review/commit/push, then pulled before execution. Do not change live environments.

Physical GPUs 4, 5 and 7 on H200-target-server are authorized; never touch another GPU, another user's process, or the existing 14B jobs. Do not run RLinf's installer wholesale: it can rewrite shared LIBERO configuration and has dependency conflicts requiring review. Do not connect to or stop existing Ray instances. No CI workflows; no force push. No training is authorized. The evaluated inference path is mapped in [docs/action-eval/INFERENCE-PATH.md](docs/action-eval/INFERENCE-PATH.md); the adapter itself lives in the RLinf fork, which is where `get_model` and the LIBERO transforms are.

README states model identity/status; docs/action-eval records the plan, provenance and blockers. Verification records require status, timestamp/timezone, commits, checkpoint/lock hashes, exact commands, exit codes, artifact paths and limitations. No completed benchmark claim without complete outcomes; errors are not task failures.

See [the handoff](docs/action-eval/README.md) and action-eval's adapter/config skills before future implementation.
