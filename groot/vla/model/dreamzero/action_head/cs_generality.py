"""Case study (CVPR'27 Sparse-rethinking), generality round (2026-11-09): DreamZero's side of FastWAM's time-structure checks.

One LIBERO call denoises one block of ``F`` (2) latent frames on a ``gh`` x ``gw`` (5 x 10) token grid per frame (agentview left half,
wrist right half; tokens of 2 x 2 latent pixels) jointly with the actions, on 16 UniPC steps, the DiT run on the ``dit_step_mask``
steps (CFG: cond + uncond). Frame 0 of the block (the near frame) covers the executed actions (frame stride 3).

``cs_jump`` (the JUMP-sparse arm, no dense step): step 0 runs the DiT with no block video key (action / state rows only; every video
token stays on noise, x^0 = 0); step 1 with the near-frame set ``s1`` (key + log(N / n)); its clean estimate's change from the current
frame x ``gain`` spread over every token (near frame: block-local inverse distance over the 4 nearest computed tokens of the same view;
far frame: near + ``far_lambda`` (near - current)) is the draft; the steps in between run nothing (video along the draft, action along
step 1's velocity); the last step runs the DiT with the near-frame half ``last`` (key + log(N / n)), its tokens' velocity replacing the
draft's.

``cs_gen`` (dense diagnostics, beside a normal dense call): F(s); step 1's sparse forwards of the strat / near sets (kappa_1 against the
dense step, late-layer self / neighbour shares, and the all-token SDPA forward against the pass); c(s) / c_grip(s) by continuing the
call from step s's recorded state (UniPC states copied) with the action clean estimate's executed xyz scaled by 1 + eps (about the
normalized 0) or its gripper + g.
"""

from __future__ import annotations

import copy
import math

import torch
from einops import rearrange

PATCH = 2  # DiT patch (1, 2, 2)


def tokens(lat: torch.Tensor) -> torch.Tensor:
    """[B, F, C, H, W] latents -> [F * gh * gw, C * 4] tokens (batch 0, fp32)."""
    return rearrange(lat[0].float(), "f c (h y) (w z) -> (f h w) (c y z)", y=PATCH, z=PATCH)


def untokens(tok: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Inverse of ``tokens`` into the layout / dtype of ``like`` [B, F, C, H, W] (fp32 kept)."""
    f, c, h, w = like.shape[1:]
    return rearrange(tok, "(f h w) (c y z) -> f c (h y) (w z)", f=f, h=h // PATCH, w=w // PATCH, c=c, y=PATCH, z=PATCH)[None]


def geometry(n_frames: int, gh: int, gw: int):
    """Per token: (frame, row, col, view) LongTensors."""
    j = torch.arange(n_frames * gh * gw)
    rem = j % (gh * gw)
    return j // (gh * gw), rem // gw, rem % gw, (rem % gw) // (gw // 2)


def strat_set(n: int, n_frames: int, gh: int, gw: int, seed: int, frames: tuple = None) -> torch.Tensor:
    """``n`` tokens stratified over the (frame, view) blocks of ``frames`` (all by default), as equal as possible (earlier blocks take
    the remainder), uniform within a block (torch generator ``seed``); sorted."""
    fr, _, _, view = geometry(n_frames, gh, gw)
    frames = tuple(range(n_frames)) if frames is None else frames
    blocks = [torch.nonzero((fr == f) & (view == v)).flatten() for f in frames for v in (0, 1)]
    per = [n // len(blocks) + (1 if i < n % len(blocks) else 0) for i in range(len(blocks))]
    g = torch.Generator().manual_seed(int(seed))
    return torch.sort(torch.cat([b[torch.randperm(len(b), generator=g)[:k]] for b, k in zip(blocks, per)])).values


def spread(zf: torch.Tensor, rows: torch.Tensor, z0: torch.Tensor, n_frames: int, gh: int, gw: int, far_lambda: float,
           k: int = 4) -> torch.Tensor:
    """Draft of every token [T, D] from the computed near-frame ``rows`` (values ``zf[rows]``): a near-frame token not computed takes
    the inverse-distance mean of the ``k`` nearest computed tokens of its view; the far frames = near + far_lambda (near - z0)."""
    fr, hh, ww, view = geometry(n_frames, gh, gw)
    per = gh * gw
    out = z0.clone()
    rows = rows.cpu()
    out[rows] = zf[rows]
    for t in range(per):
        if bool((rows == t).any()):
            continue
        src = rows[(view[rows] == view[t]) & (fr[rows] == 0)]
        if not src.numel():
            continue
        d = ((hh[src] - hh[t]).float() ** 2 + (ww[src] - ww[t]).float() ** 2).sqrt()
        o = torch.argsort(d, stable=True)[:k]
        wgt = 1.0 / d[o]
        out[t] = (wgt[:, None].to(zf.device) * zf[src[o]]).sum(dim=0) / float(wgt.sum())
    near = out[:per]
    for f in range(1, n_frames):
        out[f * per:(f + 1) * per] = near + far_lambda * (near - z0[:per])
    return out


def pixel_mask(rows: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Bool latents mask [B, F, C, H, W] of the tokens ``rows``."""
    t = torch.zeros(like.shape[1] * (like.shape[3] // PATCH) * (like.shape[4] // PATCH), like.shape[2] * PATCH * PATCH,
                    dtype=torch.bool, device=like.device)
    t[rows.to(like.device)] = True
    return untokens(t, like).bool()


class JumpState:
    """The JUMP-sparse arm across the steps of one call."""

    def __init__(self, cfg: dict, n_frames: int, gh: int, gw: int):
        self.cfg, self.n_frames, self.gh, self.gw = cfg, n_frames, gh, gw
        per = gh * gw
        self.s1 = strat_set(int(cfg.get("s1", 40)), n_frames, gh, gw, int(cfg.get("seed", 1234)), frames=(0,))
        self.last = strat_set(int(cfg.get("last", per // 2)), n_frames, gh, gw, int(cfg.get("seed", 1234)) + 1, frames=(0,))
        self.draft = None
        self.act = None
        self.runs: list = []

    def select(self, index: int, n_steps: int):
        """The block video keys of the DiT on step ``index`` (None: no DiT on this step)."""
        n_tok = self.n_frames * self.gh * self.gw
        if index == 0:
            return torch.zeros(0, dtype=torch.long), None
        rows = self.s1 if index == 1 else (self.last if index == n_steps - 1 else None)
        if rows is None:
            return None, None
        return rows, torch.full((int(rows.numel()),), math.log(n_tok / int(rows.numel())))

    def video_flow(self, index: int, n_steps: int, sample: torch.Tensor, sigma: float, model_flow, z0: torch.Tensor) -> torch.Tensor:
        """The video velocity of step ``index`` [B, F, C, H, W] (fp32 -> sample dtype)."""
        if index == 0:
            return (sample.float() / sigma).to(sample.dtype)
        if index == 1:
            x0 = tokens(sample.float() - sigma * model_flow.float())
            z0t = tokens(z0).repeat(self.n_frames, 1) if z0.shape[1] == 1 else tokens(z0)
            g = float(self.cfg.get("gain", 1.0))
            zf = z0t.clone()
            zf[self.s1] = z0t[self.s1] + g * (x0[self.s1] - z0t[self.s1])
            d = spread(zf, self.s1, z0t, self.n_frames, self.gh, self.gw, float(self.cfg.get("far_lambda", 0.5)))
            self.draft = untokens(d, sample).to(sample.device)
        v = (sample.float() - self.draft) / sigma
        if index == n_steps - 1:
            m = pixel_mask(self.last, sample)
            v = torch.where(m, model_flow.float(), v)
        return v.to(sample.dtype)


def kappa(xs: torch.Tensor, y: torch.Tensor, z0: torch.Tensor, rows: torch.Tensor) -> float:
    """sum <x - z0, y - z0> / sum |y - z0|^2 over the tokens ``rows`` (token layout [T, D])."""
    dy = y[rows] - z0[rows]
    return float(((xs[rows] - z0[rows]) * dy).sum() / (dy * dy).sum().clamp_min(1e-12))


def f_curve(rec: list, z0t: torch.Tensor, n_frames: int, gh: int, gw: int) -> dict:
    """F(s) per token group from the recorded (sample, sigma, video velocity) of every step."""
    fr, _, _, view = geometry(n_frames, gh, gw)
    groups = {"all": torch.ones_like(fr, dtype=torch.bool), "near": fr == 0, "far": fr > 0, "agent": view == 0, "wrist": view == 1,
              "near_agent": (fr == 0) & (view == 0), "near_wrist": (fr == 0) & (view == 1), "far_agent": (fr > 0) & (view == 0),
              "far_wrist": (fr > 0) & (view == 1)}
    dd = [tokens(r["v"].float() - r["sig_v"] * r["fv"].float()) - z0t for r in rec]
    den = (dd[-1] ** 2).sum(dim=-1)
    out = {}
    for name, m in groups.items():
        m = m.to(den.device)
        out[name] = [round(float(((dd[s] * dd[-1]).sum(dim=-1)[m].sum()) / den[m].sum()), 6) for s in range(len(rec))]
    return out


def copy_schedulers(sched_v, sched_a):
    return copy.deepcopy(sched_v), copy.deepcopy(sched_a)
