# Dynamic M1/M2 quality report

Date: 2026-08-30

## Status

This report records preliminary quality gates only. Full video evaluation,
worst-case task coverage, at least 100 paired requests, and closed-loop success
non-inferiority remain incomplete. No final quality claim is made.

## Full-budget invariants

The checkpoint validation path continues to require full-budget video, action,
and every-layer KV agreement with Dense. Sparse results may not reduce the
fixed eight real DiT evaluations inside the released 16-step scheduler.

## Balanced-policy eight-DiT smoke

One warmup and three paired measured real WebSocket requests were run with the
same generated observations, prompt, and seed. Dense used GPUs 2--3 and the
dynamic balanced Packed M2 policy used GPUs 5--6. Every request executed eight
real DiT evaluations.

| Metric | Result | Gate | Status |
| --- | ---: | ---: | --- |
| action cosine mean | 0.999634 | >= 0.999 | pass |
| action cosine minimum | 0.999485 | >= 0.999 | pass |
| action relative L2 mean | 2.696% | <= 5% | pass |
| action relative L2 maximum | 3.292% | <= 5% | pass |
| paired requests | 3 | >= 100 final | preliminary |

The worst measured action request was request index 1. Its action cosine was
0.999485 and relative L2 was 3.292%; it is retained in `comparison.json`
rather than hidden by the mean.

The service response exposes action but not the generated video tensor, so the
earlier single-timestep video gates are not replaced by this result. A full
trajectory video artifact/comparison path and closed-loop task evaluation are
still required.

Artifacts:

`dynamic_m1_m2/e2e/20260830_balanced_smoke/`
