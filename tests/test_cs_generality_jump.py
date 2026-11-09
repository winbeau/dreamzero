"""CPU checks of the case study's JumpState switches (dz-cause, 2026-11-09): defaults = generality's JN arms; anchor dense, extra, mode r."""

import math

import pytest
import torch

from groot.vla.model.dreamzero.action_head import cs_generality as cg

F, GH, GW = 2, 5, 10
N = F * GH * GW
JN105 = {"s1": 40, "last": 50, "seed": 1234, "far_lambda": 0.5, "gain": 1.05}


def _sched(js, n=16):
    out = []
    for i in range(n):
        rows, bias = js.select(i, n)
        out.append(None if rows is None else ("all" if isinstance(rows, str) else int(rows.numel())))
    return out


def test_defaults_are_the_jn_arm():
    js = cg.JumpState(dict(JN105), F, GH, GW)
    assert js.mode == "jump" and js.anchor == "near" and js.extra == [] and not js.native
    assert _sched(js) == [0, 40] + [None] * 13 + [50]
    rows, bias = js.select(15, 16)
    assert torch.equal(rows, js.last) and torch.allclose(bias, torch.full((50,), math.log(2.0)))


def test_extra_and_dense_anchor_schedules():
    s2 = cg.JumpState({**JN105, "extra": [14]}, F, GH, GW)
    assert _sched(s2) == [0, 40] + [None] * 12 + [50, 50]
    d2 = cg.JumpState({**JN105, "anchor": "dense", "gain": 1.0, "extra": [14]}, F, GH, GW)
    assert _sched(d2) == [0, "all"] + [None] * 12 + [50, 50]
    assert cg.JumpState({"mode": "r", "last": 50}, F, GH, GW).native
    for bad in ({"mode": "x"}, {"anchor": "x"}):
        with pytest.raises(ValueError):
            cg.JumpState(bad, F, GH, GW)


def test_dense_anchor_draft_is_the_clean_estimate_and_extra_reads_the_model():
    g = torch.Generator().manual_seed(0)
    sample = torch.randn(1, F, 4, 2 * GH, 2 * GW, generator=g)
    flow = torch.randn(1, F, 4, 2 * GH, 2 * GW, generator=g)
    z0 = torch.randn(1, 1, 4, 2 * GH, 2 * GW, generator=g)
    sig = 0.9
    d = cg.JumpState({**JN105, "anchor": "dense", "gain": 1.0, "extra": [14], "record_draft": True}, F, GH, GW)
    d.video_flow(1, 16, sample, sig, flow, z0)
    assert torch.allclose(d.draft, sample - sig * flow, atol=1e-6)  # every token's own x^0, no spread / extrapolation
    assert d.rec["shape"] == [GH * GW, 16] and d.rec["draft_near_f16"] and d.rec["z0_near_f16"]
    flow2 = torch.randn(1, F, 4, 2 * GH, 2 * GW, generator=g)
    v = d.video_flow(14, 16, sample, 0.2, flow2, z0)
    m = cg.pixel_mask(d.last, sample)
    assert torch.equal(v[m], flow2[m]) and torch.allclose(v[~m], ((sample - d.draft) / 0.2)[~m])
    v7 = d.video_flow(7, 16, sample, 0.5, None, z0)  # a skipped step: along the draft everywhere
    assert torch.allclose(v7, (sample - d.draft) / 0.5)
    s = cg.JumpState(dict(JN105), F, GH, GW)
    s.video_flow(1, 16, sample, sig, flow, z0)
    assert not s.rec and not torch.allclose(s.draft, sample - sig * flow)  # the near arm spreads from its 40 tokens
