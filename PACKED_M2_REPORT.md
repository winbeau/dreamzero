# Packed M2 report

Date: 2026-08-31

## Status

The fixed-budget Packed Middle Stack is integrated into the real DreamZero-14B
AR execution path and passes a released-checkpoint H200 gate. With both
historical K/V and current Q/K/V/compute at 20%, it reaches 2.42x--2.82x real
single-DiT speedup when only one or three Dense layers are retained at each
boundary. This exceeds the 1.80x paper target and 2.10x stretch target for the
DiT kernel path.

This is not yet the main quality result. Fixed 20% strongly changes the video
output, and the current executor still uses one budget shared by all layers and
heads. The dynamic timestep/layer/head policy, grouped-head kernels,
preallocated workspaces, eight-DiT paired benchmark, end-to-end benchmark, and
closed-loop gate remain open.

Implementation commits:

- `3aff1ef`, `c98d386`: nested, view-balanced packed-route primitives;
- `2a7d383`: one shared packed timestep-modulation gather;
- `fb3ace1`: complete packed self-attention, cross-attention, and FFN block;
- `24bcf4a`: removed packed hot-path host synchronization;
- `bed1404`: integrated one-gather/multi-layer/one-recovery execution into the
  40-layer DreamZero model;
- `1921eaa`, `9a7137c`: block-level released-path equivalence tests;
- `d051d70`: matched the released action/state RoPE storage layout;
- `da049a4`: real-checkpoint boundary scan and explicit packed-shape metrics.

All listed commits are pushed to `origin/codex/dreamzero-anchor-sparse-opt` and
the H200 checkout is fast-forwarded through them.

## Executor invariants implemented

- Prefix and suffix blocks are genuinely Dense in packed mode.
- Layer zero computes action-conditioned route scores without gathering a
  sparse historical route.
- Current video tokens and all 25 action/state registers are gathered exactly
  once after the Dense prefix.
- The middle stack keeps the activation and six-way timestep modulation packed
  across layers.
- Q/K/V/O, self-attention, cross-attention, and FFN execute only on the packed
  current sequence.
- Historical K/V uses a nested profile relative to the exact Dense history
  window; reducing current Q length cannot expose additional older KV tokens.
- The route/profile and per-layer gathered immutable historical KV are reused
  across denoising calls for the same rollout.
- Current and historical lower-budget indices are strict prefixes of higher
  budgets and remain balanced across wrist/left/right views.
- Packed RoPE uses each token's original video frame/row/column or action/state
  position. The released checkpoint's `[L,1,D]` video and `[L,D]` register RoPE
  layouts are both covered by tests.
- A full-budget request disables the packed executor and uses the original
  Dense path.
- Scatter back to the full sequence occurs only at the Dense suffix/output
  boundary. Per-layer spatial propagation is rejected in packed mode.

The first implementation uses one shared head group and allocates packed
tensors once per DiT. Multiple fixed-shape head groups and persistent buffer
preallocation are intentionally left for the dynamic M1 integration rather
than approximated with one kernel per head.

## Test gates

The H200 CPU/unit group passes 44 tests, including:

- complete and unique nested prefixes;
- minimum-view coverage;
- exact pack/recover;
- original-position RoPE equivalence;
- explicit Dense-history-window preservation;
- Q/K/V projection length restricted to the effective packed sequence;
- complete packed block equivalence at full budget;
- post-checkpoint packed configuration and Dense fallback behavior;
- M1 and server benchmark regression tests.

The real DreamZero-DROID checkpoint gate confirms that full-budget video,
action, and every returned layer KV cache are exactly equal to Dense on both
H200 ranks.

## Fixed-20% real H200 results

Protocol:

- released DreamZero-DROID 14B AR checkpoint;
- BF16 FlashAttention 2, eager execution;
- seven history frames and two current frames;
- 40 layers, 40 heads, hidden width 5120;
- 24 action tokens plus one state token always Dense;
- no KV-cache mutation during the timed action-denoise forward;
- same real DiT count; no scheduler or DiT skipping;
- one warmup followed by five measured forwards for the boundary scan.

Packed attention uses 352 current video tokens plus 25 registers as queries,
and 1,232 historical video tokens plus the same 352 current tokens and 25
registers as keys/values. Thus the fixed-shape call is 377 queries by 1,609
keys/values, versus 1,785 by 7,945 for Dense.

| Dense prefix/suffix | Dense p50 | Packed p50 | DiT speedup | Action cosine | Action rel-L2 | Video rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 | 188.04 ms | 66.65 ms | 2.821x | 0.999868 | 1.762% | 224.47% |
| 3 / 3 | 190.37 ms | 78.67 ms | 2.420x | 0.999846 | 1.856% | 127.25% |
| 5 / 5 | 192--193 ms | 112--115 ms | 1.686--1.717x | pending in v1 gate | 1.936% | 93.50% |

The 1/1 and 3/3 candidates pass the action cosine >=0.999 and action relative
L2 <=5% gates on this deterministic checkpoint input. They do not pass a video
quality gate. These numbers therefore establish executor speed and action-path
viability, not task-level non-inferiority.

Peak allocated memory is 48.19 GiB per rank. The packed path currently reduces
compute latency rather than checkpoint residency, because the full 14B weights
and immutable Dense KV cache remain resident.

Artifacts:

```text
/data/chenjiayu/wenbiao_zhao/dreamzero-anchor-sparse-artifacts/
  dynamic_m1_m2/packed_m2/20260830_fixed20_gpu01_smoke_v2/
  dynamic_m1_m2/packed_m2/20260830_fixed20_boundary_gpu01/
```

The initial smoke directory is retained as a failed engineering trace: it
exposed the real RoPE rank mismatch before `d051d70`. No failed run is included
in the performance table.

## Interpretation

The previous 1.5x result was not a fundamental DiT limit. It kept too many
Dense boundary layers and repeatedly materialized full current activations.
Once the middle stack stays packed, reducing 38 of 40 layers yields 2.82x even
though the text/image cross-attention, registers, Dense boundaries, router,
and final head remain.

The Oracle rejects a simple "later layers are more sparse" policy: late layers
are the most sensitive region. The main dynamic design must therefore retain a
cheap packed representation while raising late-layer budgets or restoring a
small Dense suffix. The 3/3 point provides substantial speed margin for this
recovery.

## Remaining Packed M2 work

- consume a timestep/layer/head-class budget schedule rather than one ratio;
- pack at the maximum active layer budget and change only prefix lengths;
- add a small number of fixed-shape shared head groups;
- use group-specific Q/K/V/O projections so a critical group does not force all
  heads to the maximum token length;
- preallocate packed activation, RoPE, Q/K/V, historical KV, output, and restore
  buffers;
- profile router, classification, pack, projections, QK, softmax, PV,
  cross-attention, FFN, propagation, and recovery independently;
- run eight real DiT evaluations, 100 paired requests, three GPU swaps, and the
  complete WebSocket benchmark;
- validate dynamic video/action quality and closed-loop success before making a
  main-result claim.

## Heterogeneous-head checkpoint

Packed M2 now supports separate nested historical-KV and current-QKV budgets
for a small number of shared head groups. Video Q/K/V/O channels are sliced,
QKV/O weights are prepacked, all action/state registers stay Dense, and the
heterogeneous groups can share one FA2 varlen launch.

This closes the correctness question but rejects naive per-head varlen as the
paper fast path. One-head varlen batch sequences remain slower than a regular
40-head fixed-shape FA2 call. The real 14B 35% trunk reaches only 1.317x p50;
the 50% trunk reaches 1.063x p50 and 1.38x on its final warmed sample. Both
pass local action gates and fail video quality. The fixed-20% one-group result
remains the valid 2.42--2.82x executor ceiling; final grouping must be
kernel-aware and collapse unprofitable head splits to shared fixed shapes.

## Dense action-history ablation

Packed M2 can now keep the 25 action/state queries on the complete historical
K/V sequence while video queries retain the sparse historical prefix. Current
video Q/K/V/O and FFN remain packed. This adds one short 25-query FA2 call per
packed layer and is disabled by default.

At the released checkpoint geometry, the isolated attention microbenchmark
increases from 1.543 ms to 2.146 ms per layer, or about 23 ms over 38 packed
layers. The paired 14B checkpoint result is consistent but smaller after full
layer overlap: Sparse p50 increases from 154.98 to 164.43 ms. Action relative
L2 improves from 1.849% to 1.703% and cosine from 0.999832 to 0.999863; video
quality is unchanged. Full-budget video, action, and cache exactness still
pass.

The real validation replay rejects this mechanism as a standalone policy fix.
Against the same Dense actions, mean action relative L2 improves from 9.293%
to 8.377%, but zero of 18 requests pass both final-action gates and the worst
request reaches 21.06% relative L2. Mean end-to-end speedup falls from 1.476x
for the original balanced executor to 1.332x. Dense action history is retained
as a protected-action ablation and possible confidence-selected fallback, not
the shared main path.

Implementation commits: `c0ee515`, `ef52371`, and `4937d92`.

Artifacts:

```text
dynamic_m1_m2/packed_m2/20260831_dense_action_history_microbench/
dynamic_m1_m2/dynamic_budgets/20260831_dense_action_history/
dynamic_m1_m2/e2e/20260831_dense_action_history_balanced_validation18/
```

## Dynamic action-history schedule

Commit `69a32c6` replaces the all-layer boolean with an optional fixed 8x40
DiT/layer table. The table changes only whether the 25 action/state queries use
complete or sparse historical K/V; packed video shapes and the Router order do
not change. Eight standardized schedules cover no protection, all middle
layers, three layer buckets, and early/late DiT subsets.

On the early-DiT checkpoint, protecting layers 1--13 gives 154.57 ms Sparse
p50, action cosine 0.999920, and 1.515% relative L2. Protecting layers 28--38
gives 157.91 ms, 0.999842, and 1.837%. The all-layer path was slower at 164.43
ms and worse than the early-layer path at 1.703% L2. This establishes that
action-history value is layer-dependent and can be non-monotonic.

The real validation18 history chain does not validate the isolated gain.
Early-layer protection reaches 1.402x mean end-to-end speedup with paired CI95
[1.345x, 1.461x], but mean action relative L2 is 8.972%, worst L2 is 21.38%,
and zero requests pass both action gates. It remains an ablation rather than a
deployed M1 route.

The late-step checkpoint confirms that the mechanism is not merely an
early-denoise effect. At DiT index 4, early-layer protection improves action
relative L2 from 2.487% to 1.457%, but adds 13.28 ms. The remaining trajectory
failure therefore comes from state accumulated outside the protected action
readout, especially sparse video-query history and packed hidden-state error.

Artifacts:

```text
dynamic_m1_m2/dynamic_budgets/20260831_dynamic_action_history/
dynamic_m1_m2/e2e/20260831_dynamic_action_history_early_layers_validation18/
```

## Early video-history floor ablation

Raising only the first 13 packed layers' video-query historical K/V budget to
75% improves the isolated checkpoint action/video errors to 1.259%/8.525%
relative L2 without increasing current Q/K/V/O or FFN token counts. A 100%
floor is slightly worse, confirming non-monotonicity.

The complete validation18 replay reverses the local result: mean action L2 is
10.011%, worst L2 is 23.41%, and only 2/18 requests are safe, although mean
end-to-end speedup is 1.487x. This variant adds only one new safe request beyond
the conservative profile and leaves the multi-profile Oracle ceiling at
1.102x. It is retained as an ablation, not a main executor schedule.

## Dense-suffix recovery ablation

Restoring more complete output layers does reduce the isolated packed video
error: balanced suffix one, three, and five measure 8.758%, 7.512%, and 6.708%
video relative L2 at the early checkpoint.  The corresponding action errors
remain 1.849%, 1.858%, and 1.864%, showing that the suffix mostly repairs the
video representation after the action-path error has already accumulated.

Validation18 confirms the limit.  Suffix three retains 1.484x paired geometric
mean end-to-end speedup and improves mean action L2 only to 8.999%, with 1/18
safe requests and a worse 20.83% tail.  Suffix five falls to 1.420x, regresses
mean action L2 to 9.103%, and has 0/18 safe requests.  Both services execute 76
warmup/history/target calls with exactly eight DiT evaluations per call.

The final-layer recovery ablation is complete and rejected as the primary
quality mechanism.  Packed M2 should retain a small suffix and spend recovery
budget at propagation boundaries or within sensitive segments instead.

## Wider propagation recovery

The radius-two/every-five boundary was compared with radius two/every three
and radius three/every five under the same balanced packed shapes.  More
frequent radius-two recovery is neutral: checkpoint video L2 changes from
8.758% to 8.830%, action L2 from 1.849% to 1.863%, and DiT speedup remains
1.226x.  Wider radius-three recovery is locally strong, reducing video L2 to
4.233% while retaining 1.272x checkpoint DiT speedup and 1.839% action L2.

The validation18 trajectory rejects that local proxy.  Radius three/every five
is fast at 1.516x paired geometric mean end-to-end speedup, but mean action L2
is 9.553%, worst L2 is 20.289%, and zero requests are safe.  All 76 calls retain
the required eight DiT evaluations.  Because wider interpolation repairs video
without repairing the action registers, further M2 recovery must operate on
action-sensitive packed state within the segment rather than on the final
spatial field alone.

## Scheduled maximum-current K/V readout

Packed M2 now contains a bounded experiment for action/state queries to read
the maximum prepacked current-video K/V prefix while all video-token compute
stays at the active nested prefix. The implementation projects the extra K/V
only when the scheduled layer has a maximum prefix longer than its active
prefix. Original-position RoPE is used, head-group execution is rejected for
this path, and an optional 8-DiT by 40-layer boolean table controls the call.
Full-budget video, action, and every returned KV cache remain exactly equal to
Dense on both real H200 ranks. Commit `bd28485` passes 45 focused H200 tests.

The executor experiment also exposes a state-validity limit. Maximum-prefix
tokens are freshly repacked at layers 1, 6, 11, and subsequent segment
entries, but tokens outside the active prefix skip the intervening blocks.
Using those stale tail tokens at segment exits raises checkpoint action
relative L2 from 1.849% to 1.933%; enabling the readout throughout the packed
middle raises it to 2.147%. Restricting it to the eight fresh segment entries
reaches 1.845%, only a 0.21% relative improvement over no readout.

The exchanged entry/exit rounds measure virtually identical geometric-mean
DiT speedups of 1.26235x and 1.26225x. Timing against a separately loaded
`none` run is dominated by first-use compilation and run drift, so no speed
benefit is claimed. The mechanism remains implemented for reproducibility,
but it is rejected as a recovery path: a maximum packed allocation is not a
valid hidden-state substitute unless every exposed token has traversed the
same intervening layers.

Artifact:

`dynamic_m1_m2/dynamic_budgets/20260831_max_action_current/`
