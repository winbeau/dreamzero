# Dynamic M1/M2 ablation report

Date: 2026-08-31

## Status

This indexes completed and pending ablations. It is not a final claim: 100
paired requests, three GPU exchanges, closed-loop non-inferiority, accepted
confidence fallback, and a quality-safe accelerated policy remain incomplete.

## Completed executor ablations

| Ablation | Evidence | Decision |
| --- | --- | --- |
| old gather/scatter 20/20/20 | 1.429x four-GPU mean; action rel-L2 11.23% | speed baseline, reject quality |
| fixed Packed M2 20% | 2.42--2.82x DiT; large video error | performance ceiling |
| timestep/layer/joint budgets | real early/late timing changes with shape | runtime validated |
| four historical-KV head groups | 1.17x early, 1.60x late | reject multi-launch overhead |
| propagation radius 2/every 5 | video cosine 0.8987 at 1.742x | retain |
| 75% segment floor | video cosine 0.9970 at 1.378x | quality reference, too slow |
| two-group sliced QKV, two calls | 2.36 ms synthetic attention | reject |
| prepacked fused QKV/O | 2.31 ms | insufficient alone |
| one head-as-batch varlen FA2 | 2.11 ms versus 1.54 ms regular FA2 | negative ablation |
| trunk 50%, critical H100/Q50, normal H35/Q25 | action 0.999908; video 0.953; <=1.38x warmed | reject |
| trunk 35%, critical H100/Q35, normal H25/Q20 | action 0.999901; video 0.878; 1.317x | reject |
| balanced/conservative/Dense selector | held-out Oracle ceiling <=1.125x | profile family rejected |
| action-flow sentinel | test false-sparse 6.67%; triggers all safe requests | reject |
| Dense action-history microbenchmark | +0.603 ms/layer attention; about +23 ms/38 layers | protected-action overhead |
| Dense action-history validation18 | L2 9.293% -> 8.377%; 0/18 safe; 1.332x | reject global policy |
| action history layers 1--13 vs 28--38 | checkpoint L2 1.515% vs 1.837%; 154.57 vs 157.91 ms | early layers favored locally |
| early-layer action history validation18 | 1.402x; L2 8.972%; 0/18 safe | reject global schedule |
| DiT-4 early-layer action history vs none | L2 1.457% vs 2.487%; 145.59 vs 132.31 ms | quality gain, too costly alone |
| early video-history 75% vs 100% | checkpoint action L2 1.259% vs 1.360% | 75% locally favored |
| early video-history75 validation18 | 1.487x; L2 10.011%; 2/18 safe | reject global schedule |
| five-profile validation Oracle | safety union 8/18; 1.102x ratio-of-means | coarse profile family exhausted |
| Dense suffix 3 checkpoint | video L2 7.512%; action L2 1.858%; 1.252x | local video recovery |
| Dense suffix 5 checkpoint | video L2 6.708%; action L2 1.864%; 1.159x | recovery cost dominates |
| Dense suffix 3 validation18 | 1.484x; action L2 8.999%; 1/18 safe | reject global policy |
| Dense suffix 5 validation18 | 1.420x; action L2 9.103%; 0/18 safe | reject deeper suffix |
| propagation radius 2/every 3 | checkpoint video L2 8.830%; 1.226x | frequency increase ineffective |
| propagation radius 3/every 5 | checkpoint video L2 4.233%; 1.272x | strong local video recovery |
| radius 3/every 5 validation18 | 1.516x; action L2 9.553%; 0/18 safe | reject global propagation |
| max current K/V at segment entries | action L2 1.849% -> 1.845%; 1.262x exchanged geomean | negligible gain; reject |
| max current K/V at segment exits | action L2 1.933% | stale inactive state; reject |
| max current K/V at all packed layers | action L2 2.147% | stale-readout negative ablation |
| individual-cell H75/Q25, late3 | worst cosine 0.99806; L2 6.37%; 1.206x | reject current floor |
| propagation-aligned S4 H75/Q50, late3 | cosine min 0.99932; L2 max 3.87%; 1.252x | pass pilot |
| propagation-aligned S4 H75/Q35, late3 | cosine min 0.99623; L2 max 8.72%; 1.239x | reject Q35 |
| propagation-aligned S4 H50/Q50, late3 | cosine min 0.99938; L2 max 3.93%; 1.269x | pass pilot |
| timestep expansion late3 -> late4 | 1.269x -> 1.313x; L2 max 4.09% | promote validation |
| timestep expansion late4 -> late5 | 1.325x; cosine min 0.98463; L2 max 18.41% | reject early DiT 3 |
| layer expansion S4 -> S5 at late4 | 1.267x; cosine min 0.99825; L2 max 5.92% | reject fifth segment |
| late4/S4/H50Q50 validation18 | 1.074x; 14/18 safe; cosine min 0.99802; L2 max 7.51% | reject static candidate |
| late4/S4/H50Q50 full 108 requests | 1.074--1.091x by split; 92/108 safe | stable speed, reject quality |
| maximum-Head shared M1 promotion | every validation/test candidate cell Dense | safe but zero sparse coverage |
| request-wide Packed-proxy promotion gate | val/test 0 false-sparse for selected GB; train episode-CV 6/72 false-sparse; about 1.01x mixed | reject safety and speed |

## Required remaining ablations

- kernel-aware fixed-shape grouping versus heterogeneous varlen;
- fixed, timestep-only, layer-only, head-only, timestep+layer, and full
  timestep+layer+head under the same eight-DiT service protocol;
- confidence fallback on/off with an accepted deployment-safe policy;
- GMM versus supervised M1 on final executor labels;
- random, uniform, action-anchor, and Oracle routes;
- no extrapolation versus late-step VV extrapolation with sentinel;
- within-segment action-sensitive recovery that updates token state rather
  than exposing a stale maximum-prefix readout;
- 100 paired requests, three GPU exchanges, and closed-loop success.

## Artifact roots

```text
dynamic_m1_m2/baseline/
dynamic_m1_m2/packed_m2/
dynamic_m1_m2/dynamic_budgets/
dynamic_m1_m2/m1_classifier/
dynamic_m1_m2/request_gate/
dynamic_m1_m2/e2e/
```
