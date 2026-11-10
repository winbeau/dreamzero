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
    """The JUMP-sparse arm across the steps of one call.

    dz-cause (2026-11-09) switches, all off by default (= generality's JN arms): ``mode: r`` (the R arm: the native dense schedule on
    steps 0 .. last-1, only the last step reads ``last``; the loop handles it, ``native``); ``anchor: dense`` (step 1 runs the native
    DiT on every block video key, the draft = every token's clean estimate x ``gain``, no spread, no extrapolation); ``extra``
    (steps that also run the DiT on ``last``, as the last step: its tokens and the action take the model's velocity, the rest stays
    on the draft); ``record_draft`` (the near frame's draft and current-frame tokens are logged)."""

    MODES, ANCHORS = ("jump", "r"), ("near", "dense")

    def __init__(self, cfg: dict, n_frames: int, gh: int, gw: int):
        self.cfg, self.n_frames, self.gh, self.gw = cfg, n_frames, gh, gw
        per = gh * gw
        self.mode, self.anchor = str(cfg.get("mode", "jump")), str(cfg.get("anchor", "near"))
        if self.mode not in self.MODES or self.anchor not in self.ANCHORS:
            raise ValueError(f"cs_jump: mode {self.mode!r} / anchor {self.anchor!r} not in {self.MODES} / {self.ANCHORS}")
        self.extra = sorted({int(x) for x in cfg.get("extra", ())})
        self.native = self.mode == "r"
        self.s1 = strat_set(int(cfg.get("s1", 40)), n_frames, gh, gw, int(cfg.get("seed", 1234)), frames=(0,))
        self.last = strat_set(int(cfg.get("last", per // 2)), n_frames, gh, gw, int(cfg.get("seed", 1234)) + 1, frames=(0,))
        # chain2-curves (2026-11-10): the last step's near-frame set and per-key bias given explicitly (the adapter builds them from
        # the call's body mask); ``record_recv``: the attention every read key receives on the last step (sink finding)
        self._last_bias = None
        if cfg.get("last_rows") is not None:
            rows = [int(x) for x in cfg["last_rows"]]
            bias = [float(x) for x in cfg.get("last_bias") or ()]
            if not rows or len(set(rows)) != len(rows) or min(rows) < 0 or max(rows) >= per or len(bias) != len(rows):
                raise ValueError("cs_jump last_rows: distinct near-frame tokens with one last_bias each")
            order = sorted(range(len(rows)), key=lambda i: rows[i])
            self.last = torch.tensor([rows[i] for i in order], dtype=torch.long)
            self._last_bias = torch.tensor([bias[i] for i in order], dtype=torch.float32)
        self.record_recv = bool(cfg.get("record_recv", False))
        self.draft = None
        self.act = None
        self.runs: list = []
        self.rec: dict = {}

    def last_bias(self) -> torch.Tensor:
        if self._last_bias is not None:
            return self._last_bias.clone()
        n_tok = self.n_frames * self.gh * self.gw
        return torch.full((int(self.last.numel()),), math.log(n_tok / int(self.last.numel())))

    def reads_last(self, index: int, n_steps: int) -> bool:
        return index == n_steps - 1 or index in self.extra

    def select(self, index: int, n_steps: int):
        """The block video keys of the DiT on step ``index`` (None: no DiT on this step; "all": the native DiT, every key)."""
        n_tok = self.n_frames * self.gh * self.gw
        if index == 0:
            return torch.zeros(0, dtype=torch.long), None
        if index == 1:
            if self.anchor == "dense":
                return "all", None
            return self.s1, torch.full((int(self.s1.numel()),), math.log(n_tok / int(self.s1.numel())))
        if self.reads_last(index, n_steps):
            return self.last, self.last_bias()
        return None, None

    def video_flow(self, index: int, n_steps: int, sample: torch.Tensor, sigma: float, model_flow, z0: torch.Tensor) -> torch.Tensor:
        """The video velocity of step ``index`` [B, F, C, H, W] (fp32 -> sample dtype)."""
        if index == 0:
            return (sample.float() / sigma).to(sample.dtype)
        if index == 1:
            x0 = tokens(sample.float() - sigma * model_flow.float())
            z0t = tokens(z0).repeat(self.n_frames, 1) if z0.shape[1] == 1 else tokens(z0)
            g = float(self.cfg.get("gain", 1.0))
            if self.anchor == "dense":  # dz-cause D: every token's own clean estimate, no spread / extrapolation
                d = z0t + g * (x0 - z0t)
            else:
                zf = z0t.clone()
                zf[self.s1] = z0t[self.s1] + g * (x0[self.s1] - z0t[self.s1])
                d = spread(zf, self.s1, z0t, self.n_frames, self.gh, self.gw, float(self.cfg.get("far_lambda", 0.5)))
            self.draft = untokens(d, sample).to(sample.device)
            if self.cfg.get("record_draft"):
                per = self.gh * self.gw
                self.rec = {"draft_near_f16": f16_b64(d[:per]), "z0_near_f16": f16_b64(z0t[:per]), "shape": [per, int(d.shape[1])]}
        v = (sample.float() - self.draft) / sigma
        if self.reads_last(index, n_steps):
            m = pixel_mask(self.last, sample)
            v = torch.where(m, model_flow.float(), v)
        return v.to(sample.dtype)


def f16_b64(x: torch.Tensor) -> str:
    """A tensor as base64 float16 bytes (row-major), for the probe logs."""
    import base64

    return base64.b64encode(x.detach().float().cpu().half().numpy().tobytes()).decode()


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
