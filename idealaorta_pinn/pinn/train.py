"""Trainer for the parametric PINN surrogate.

Implements the group's house-style training loop:
  * decoupled per-field networks (u, v, w, p) + a turbulent-viscosity net (nut),
  * dual AdamW optimizers (nut at a higher LR) so the closure field gets signal,
  * gradient-norm adaptive loss weighting (Wang et al., 2021) with EMA + cap,
    and a manual-weight fallback,
  * per-(case, phase) accumulation of data + physics + BC losses,
  * checkpoints carrying the networks, the fitted Normalizer, and ref scales.

Driven by a plain config dict (loaded from a stage YAML by scripts/03_train.py).
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

from ..config import MODELS_DIR, PROJECT_ROOT
from ..data.loaders import Bundle, build_bundle, fit_normalizer, load_holdout_velocity
from ..data.normalize import Normalizer
from ..data.registry import load_registry
from .losses import (data_pressure_loss, data_velocity_loss, inlet_velocity_loss,
                     noslip_loss, outlet_pressure_loss, relative_l2, wss_loss)
from .model import count_parameters, create_networks
from .physics import compute_physics_loss, compute_residuals

COMPONENTS = ["velocity", "physics", "pressure", "wss", "noslip", "inlet", "outlet"]


class Trainer:
    def __init__(self, config: Dict):
        self.cfg = config
        self.name = config["experiment"]["name"]
        # Phase-balanced velocity supervision: relative (per-group) velocity loss
        # so the near-stagnant diastolic phase is not swamped by systolic-scale
        # magnitudes under the single U_ref (off by default; opt-in per config).
        self.vel_relative = bool(config.get("loss_balance", {}).get("relative_velocity", False))
        # S1: per-phase velocity scaling — geometry/physics scale stays systolic;
        # the velocity nets emit O(1) for both phases and the data-loss budget is
        # equalized per phase. Supersedes relative_velocity (don't enable both).
        self.per_phase_vel = bool(config.get("loss_balance", {}).get("per_phase_velocity_scale", False))
        # R1/R2: one seed, propagated to torch (CPU+CUDA), numpy, AND the data
        # pipeline (fit_normalizer / build_bundle below), plus deterministic kernels,
        # so changing random_seed actually changes the whole run reproducibly.
        self.seed = int(config.get("random_seed", 42))
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        if bool(config.get("deterministic", False)):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.device = config.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[trainer] CUDA unavailable -> CPU")
            self.device = "cpu"

        # Output dir is anchored to the project root (never the working dir) so
        # results always land inside the repo, wherever the script is launched.
        od = Path(config["output_dir"]) if config.get("output_dir") else MODELS_DIR / self.name
        self.out_dir = od if od.is_absolute() else PROJECT_ROOT / od
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._build_data()
        self._build_networks()
        self._build_optimizers()
        self._build_batched()

        self.epoch = 0
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.no_improve = 0
        self.history: List[Dict] = []
        self.adaptive_weights: Dict[str, float] = {}
        # Crash recovery: if config.resume and a resume_state.pt exists in out_dir,
        # restore full training state (nets, both optimizers/schedulers, epoch,
        # best/patience, adaptive weights, history, RNG) and continue. The data
        # pipeline is seed-deterministic so the rebuilt normalizer/bundle match.
        self._maybe_resume()

    # ------------------------------------------------------------------ data
    def _build_data(self) -> None:
        data = self.cfg["data"]
        phys = self.cfg.get("physics", {})
        mu = float(phys.get("mu", 0.0035))
        rho = float(phys.get("rho", 1060.0))
        records = load_registry()
        self.records = records

        train_cases = list(data["train_cases"])
        phases = list(data.get("phases", ["systolic", "diastolic"]))
        velocity_kinds = tuple(data.get("velocity_kinds", ["3D"]))

        print(f"[trainer] fitting normalizer on cases {train_cases}, phases {phases}")
        print(f"[trainer] velocity supervision sources: {list(velocity_kinds)}")
        self.normalizer: Normalizer = fit_normalizer(
            records, train_cases, phases, mu=mu, rho=rho, seed=self.seed,
            per_phase_velocity_scale=self.per_phase_vel,
            velocity_kinds=velocity_kinds)
        if phys.get("re_override"):
            self._Re = float(phys["re_override"])
        else:
            self._Re = self.normalizer.Re
        print(f"[trainer] U_ref={self.normalizer.U_ref:.4f} m/s  L={self.normalizer.L:.4f} m  "
              f"Re={self._Re:.1f}  wss_std={self.normalizer.wss_std:.4f}")
        if self.per_phase_vel and self.normalizer.u_ref_diastolic is not None:
            print(f"[trainer] S1 per-phase velocity scaling ON: "
                  f"U_ref_diastolic={self.normalizer.u_ref_diastolic:.4f} m/s, "
                  f"diastolic gain s={self.normalizer.vel_phase_gain('diastolic'):.4f}")

        ld = self.cfg.get("loaders", {})
        self.bundle: Bundle = build_bundle(
            records, train_cases, phases, self.normalizer, device=self.device,
            max_velocity_points=int(ld.get("max_velocity_points", 40_000)),
            n_collocation=int(self.cfg.get("physics", {}).get("n_collocation", 8_000)),
            wall_normals_method=ld.get("wall_normals_method", "auto"),
            inlet_n_radial=int(ld.get("inlet_n_radial", 6)),
            inlet_n_angular=int(ld.get("inlet_n_angular", 12)),
            velocity_kinds=velocity_kinds,
            volumetric_collocation=bool(ld.get("volumetric_collocation", False)),
            seed=self.seed,
        )
        if bool(ld.get("volumetric_collocation", False)):
            n_coll = [int(g.tensors["cx"].shape[0]) for g in self.bundle.groups if "cx" in g.tensors]
            print(f"[trainer] S2 volumetric collocation ON: "
                  f"{sum(n_coll)} interior points across {len(n_coll)} groups")
        print(f"[trainer] built {len(self.bundle.groups)} (case,phase) groups")

        # Optional held-out velocity slice for validation (Stage A de-risk).
        self.holdout = None
        ho = data.get("holdout")
        if ho:
            self.holdout = load_holdout_velocity(
                records, int(ho["case"]), ho["phase"], ho["kind"],
                self.normalizer, device=self.device)
            print(f"[trainer] holdout: case {ho['case']} {ho['phase']} {ho['kind']} "
                  f"({0 if self.holdout is None else self.holdout['x'].shape[0]} pts)")

    # -------------------------------------------------------------- networks
    def _build_networks(self) -> None:
        m = self.cfg["model"]
        nut = m.get("nut", {})
        # S4: anisotropic Fourier — give each spatial axis a bandwidth matched to
        # its standardized span, so the transverse directions (short, fast-varying)
        # are not under-resolved by an isotropic basis tuned to the axial extent.
        fourier_scale = float(m.get("fourier_scale", 1.0))
        if m.get("anisotropic_fourier", False):
            spans = []
            for ax in ("vx", "vy", "vz"):
                vals = [g.tensors[ax] for g in self.bundle.groups if ax in g.tensors]
                spans.append(float((torch.cat(vals).max() - torch.cat(vals).min())) if vals else 1.0)
            spans = np.asarray(spans)
            aniso = spans.max() / np.maximum(spans, 1e-6)        # axial=1, transverse>1
            fourier_scale = (fourier_scale * aniso).tolist()
            print(f"[trainer] S4 anisotropic Fourier: per-axis scale "
                  f"{[round(s, 2) for s in fourier_scale]} (std spans {spans.round(3).tolist()})")
        self.networks = create_networks(
            n_param=self.bundle.n_param,
            hidden_dim=int(m.get("hidden_dim", 128)),
            num_layers=int(m.get("num_layers", 6)),
            num_frequencies=int(m.get("num_frequencies", 16)),
            fourier_scale=fourier_scale,
            use_fourier=bool(m.get("use_fourier", True)),
            use_param_encoder=bool(m.get("use_param_encoder", True)),
            param_encoder_dims=m.get("param_encoder_dims"),
            coord_encoder_dims=m.get("coord_encoder_dims"),
            nut_hidden_dim=int(nut.get("hidden_dim", 64)),
            nut_num_layers=int(nut.get("num_layers", 4)),
            nu_t_min=float(nut.get("nu_t_min", 1e-3)),
            initial_nut=float(nut.get("initial_nut", 0.05)),
            vel_phase_gain=(self.normalizer.vel_phase_gain("diastolic")
                            if self.per_phase_vel and self.normalizer.u_ref_diastolic is not None
                            else None),
            device=self.device,
        )
        total = sum(count_parameters(n) for n in self.networks.values())
        print(f"[trainer] networks: {total:,} params "
              f"({', '.join(f'{k}:{count_parameters(v):,}' for k, v in self.networks.items())})")

    # ------------------------------------------------------------ optimizers
    def _build_optimizers(self) -> None:
        tr = self.cfg["training"]
        lr = float(tr.get("lr", 1e-4))
        nut_mult = float(self.cfg["model"].get("nut", {}).get("lr_multiplier", 10.0))
        flow_params = [p for k, n in self.networks.items() if k != "nut" for p in n.parameters()]
        self.opt = optim.AdamW(flow_params, lr=lr, betas=(0.9, 0.99), eps=1e-12, weight_decay=1e-4)
        self.opt_nut = optim.AdamW(self.networks["nut"].parameters(), lr=lr * nut_mult,
                                   betas=(0.9, 0.99), eps=1e-12, weight_decay=1e-5)
        epochs = int(tr.get("epochs", 5000))
        eta_min = float(tr.get("scheduler", {}).get("eta_min", 1e-6))
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        self.sched_nut = optim.lr_scheduler.CosineAnnealingLR(self.opt_nut, T_max=epochs, eta_min=eta_min)
        self.grad_clip = float(tr.get("gradient_clip", 1.0))

    # ----------------------------------------------------------------- losses
    # The total loss is the unweighted MEAN over (case,phase) groups of each
    # component. Looping groups in Python (one forward/backward per group) leaves the
    # GPU ~30% utilized and dominates wall-clock once there are many cases. The hot
    # path below (_component_losses) instead runs ONE batched forward/grad over the
    # concatenated points of every group, then takes a vectorized per-group mean
    # (segment-mean via index_add_). It is mathematically identical to the readable
    # per-group reference _component_losses_reference; tests/test_batched_losses.py
    # asserts the two agree on a real bundle (both phases, S1 on).
    #
    # C1 reminder: with S1 the velocity nets emit u_s = q*s (s = vel_phase_gain), so
    # every velocity-derived term (velocity, physics, wss, noslip, inlet) is divided
    # by the group's s**2 (pg2) for an equal per-phase gradient budget; pressure +
    # outlet (P_ref-scaled, not velocity-gained) are not. s=1 for systole.
    def _build_batched(self) -> None:
        """Precompute concatenated-across-groups tensors + per-point group ids and
        per-group scales, so each loss term is one batched op instead of a loop."""
        if self.vel_relative:
            raise NotImplementedError(
                "batched loss path does not support legacy relative_velocity; "
                "use per_phase_velocity_scale (S1) instead.")
        dev = self.device

        def cat(groups, key):
            return torch.cat([g.tensors[key] for g in groups], dim=0)

        def gid_of(groups, key):
            sizes = [g.tensors[key].shape[0] for g in groups]
            idx = torch.cat([torch.full((s,), i, dtype=torch.long, device=dev)
                             for i, s in enumerate(sizes)])
            return idx, len(groups)

        def pg2(groups):
            return torch.tensor(
                [self.normalizer.vel_phase_gain(g.phase) ** 2 for g in groups],
                dtype=torch.float32, device=dev)

        B: Dict[str, dict] = {}
        vg = [g for g in self.bundle.groups if "vx" in g.tensors]
        if vg:
            gid, ng = gid_of(vg, "vx")
            B["vel"] = dict(x=cat(vg, "vx"), y=cat(vg, "vy"), z=cat(vg, "vz"),
                            params=cat(vg, "v_params"), ut=cat(vg, "u_t"),
                            vt=cat(vg, "v_t"), wt=cat(vg, "w_t"), gid=gid, ng=ng, pg2=pg2(vg))
            cgid, cng = gid_of(vg, "cx")
            B["coll"] = dict(x=cat(vg, "cx"), y=cat(vg, "cy"), z=cat(vg, "cz"),
                             params=cat(vg, "c_params"), gid=cgid, ng=cng, pg2=pg2(vg))
        wg = [g for g in self.bundle.groups if "wx" in g.tensors]
        if wg:
            gid, ng = gid_of(wg, "wx")
            B["wall"] = dict(x=cat(wg, "wx"), y=cat(wg, "wy"), z=cat(wg, "wz"),
                             params=cat(wg, "w_params"), pt=cat(wg, "p_t"),
                             wss_t=cat(wg, "wss_t"), normals=cat(wg, "normals"),
                             gid=gid, ng=ng, pg2=pg2(wg))
            ig = [g for g in wg if "inlet_x" in g.tensors]
            if ig:
                igid, ing = gid_of(ig, "inlet_x")
                tu, tv, tw = [], [], []
                for g in ig:
                    n = g.tensors["inlet_x"].shape[0]
                    ax = int(g.meta.get("axial_dim", 0))
                    u_in = float(g.meta.get("u_inlet_nd", 0.0))
                    cols = [torch.zeros(n, 1, device=dev) for _ in range(3)]
                    cols[ax] = torch.full((n, 1), u_in, device=dev)
                    tu.append(cols[0]); tv.append(cols[1]); tw.append(cols[2])
                B["inlet"] = dict(x=cat(ig, "inlet_x"), y=cat(ig, "inlet_y"), z=cat(ig, "inlet_z"),
                                  params=cat(ig, "inlet_params"), tu=torch.cat(tu),
                                  tv=torch.cat(tv), tw=torch.cat(tw), gid=igid, ng=ing, pg2=pg2(ig))
            og = [g for g in wg if "outlet_x" in g.tensors]
            if og:
                ogid, ong = gid_of(og, "outlet_x")
                B["outlet"] = dict(x=cat(og, "outlet_x"), y=cat(og, "outlet_y"),
                                   z=cat(og, "outlet_z"), params=cat(og, "outlet_params"),
                                   gid=ogid, ng=ong)
        self._bz = B

    @staticmethod
    def _seg_mean(per_point: torch.Tensor, gid: torch.Tensor, ng: int) -> torch.Tensor:
        """Per-group mean of a per-point quantity (vectorized; replaces a group loop)."""
        v = per_point.reshape(-1)
        s = torch.zeros(ng, device=v.device, dtype=v.dtype).index_add_(0, gid, v)
        c = torch.zeros(ng, device=v.device, dtype=v.dtype).index_add_(0, gid, torch.ones_like(v))
        return s / c.clamp_min(1.0)

    def _uvw(self, x, y, z, params):
        net_in = torch.cat([x, y, z, params], dim=1)
        return (self.networks["u"](net_in).view(-1, 1),
                self.networks["v"](net_in).view(-1, 1),
                self.networks["w"](net_in).view(-1, 1))

    def _physics_batched(self, c: dict) -> torch.Tensor:
        x = c["x"].clone().detach().requires_grad_(True)
        y = c["y"].clone().detach().requires_grad_(True)
        z = c["z"].clone().detach().requires_grad_(True)
        r = compute_residuals(self.networks, x, y, z, c["params"], self._Re)
        pp = r["res_x"] ** 2 + r["res_y"] ** 2 + r["res_z"] ** 2 + r["res_cont"] ** 2
        return (self._seg_mean(pp, c["gid"], c["ng"]) / c["pg2"]).mean()

    def _wss_noslip_pp(self, wb: dict):
        """Per-point (WSS sq-error meaned over 3 comps, u^2+v^2+w^2) at wall points."""
        x = wb["x"].clone().detach().requires_grad_(True)
        y = wb["y"].clone().detach().requires_grad_(True)
        z = wb["z"].clone().detach().requires_grad_(True)
        u, v, w = self._uvw(x, y, z, wb["params"])
        one = torch.ones_like(u)

        def grad(o, i):
            return torch.autograd.grad(o, i, grad_outputs=one, create_graph=True, retain_graph=True)[0]

        u_x, u_y, u_z = grad(u, x), grad(u, y), grad(u, z)
        v_x, v_y, v_z = grad(v, x), grad(v, y), grad(v, z)
        w_x, w_y, w_z = grad(w, x), grad(w, y), grad(w, z)
        txx, tyy, tzz = 2.0 * u_x, 2.0 * v_y, 2.0 * w_z
        txy, txz, tyz = (u_y + v_x), (u_z + w_x), (v_z + w_y)
        nx, ny, nz = wb["normals"][:, 0:1], wb["normals"][:, 1:2], wb["normals"][:, 2:3]
        tx = txx * nx + txy * ny + txz * nz
        ty = txy * nx + tyy * ny + tyz * nz
        tz = txz * nx + tyz * ny + tzz * nz
        tdn = tx * nx + ty * ny + tz * nz
        ws = self.normalizer.wss_std
        ex = (tx - tdn * nx) / ws - wb["wss_t"][:, 0:1]
        ey = (ty - tdn * ny) / ws - wb["wss_t"][:, 1:2]
        ez = (tz - tdn * nz) / ws - wb["wss_t"][:, 2:3]
        wss_pp = (ex ** 2 + ey ** 2 + ez ** 2) / 3.0     # _MSE over (N,3) -> mean over comps
        return wss_pp, u ** 2 + v ** 2 + w ** 2

    def _component_losses(self) -> Dict[str, torch.Tensor]:
        """Batched, math-identical reimplementation of _component_losses_reference."""
        dev = self.device
        acc = {c: torch.zeros((), device=dev) for c in COMPONENTS}
        b = self._bz.get("vel")
        if b is not None:
            u, v, w = self._uvw(b["x"], b["y"], b["z"], b["params"])
            sq = (u - b["ut"]) ** 2 + (v - b["vt"]) ** 2 + (w - b["wt"]) ** 2
            acc["velocity"] = (self._seg_mean(sq, b["gid"], b["ng"]) / b["pg2"]).mean()
        c = self._bz.get("coll")
        if c is not None:
            acc["physics"] = self._physics_batched(c)
        wb = self._bz.get("wall")
        if wb is not None:
            p = self.networks["p"](torch.cat([wb["x"], wb["y"], wb["z"], wb["params"]], dim=1)).view(-1, 1)
            acc["pressure"] = self._seg_mean((p - wb["pt"]) ** 2, wb["gid"], wb["ng"]).mean()
            wss_pp, uvw_sq = self._wss_noslip_pp(wb)
            acc["wss"] = (self._seg_mean(wss_pp, wb["gid"], wb["ng"]) / wb["pg2"]).mean()
            acc["noslip"] = (self._seg_mean(uvw_sq, wb["gid"], wb["ng"]) / wb["pg2"]).mean()
        ib = self._bz.get("inlet")
        if ib is not None:
            u, v, w = self._uvw(ib["x"], ib["y"], ib["z"], ib["params"])
            sq = (u - ib["tu"]) ** 2 + (v - ib["tv"]) ** 2 + (w - ib["tw"]) ** 2
            acc["inlet"] = (self._seg_mean(sq, ib["gid"], ib["ng"]) / ib["pg2"]).mean()
        ob = self._bz.get("outlet")
        if ob is not None:
            p = self.networks["p"](torch.cat([ob["x"], ob["y"], ob["z"], ob["params"]], dim=1)).view(-1, 1)
            acc["outlet"] = self._seg_mean(p ** 2, ob["gid"], ob["ng"]).mean()
        return acc

    def _component_losses_reference(self) -> Dict[str, torch.Tensor]:
        """Readable per-group reference (correctness oracle for the batched path;
        NOT used in the hot loop). tests/test_batched_losses.py asserts equivalence."""
        dev = self.device
        acc = {c: torch.zeros((), device=dev) for c in COMPONENTS}
        n_vel = n_wall = 0
        for g in self.bundle.groups:
            t = g.tensors
            pg2 = self.normalizer.vel_phase_gain(g.phase) ** 2
            if "vx" in t:
                acc["velocity"] = acc["velocity"] + data_velocity_loss(
                    self.networks, t["vx"], t["vy"], t["vz"], t["v_params"],
                    t["u_t"], t["v_t"], t["w_t"], relative=self.vel_relative,
                    phase_gain=self.normalizer.vel_phase_gain(g.phase))
                pl, _ = compute_physics_loss(
                    self.networks, t["cx"], t["cy"], t["cz"], t["c_params"], self._Re)
                acc["physics"] = acc["physics"] + pl / pg2
                n_vel += 1
            if "wx" in t:
                acc["pressure"] = acc["pressure"] + data_pressure_loss(
                    self.networks["p"], t["wx"], t["wy"], t["wz"], t["w_params"], t["p_t"])
                wl, _ = wss_loss(self.networks, t["wx"], t["wy"], t["wz"], t["w_params"],
                                 t["wss_t"], t["normals"], self.normalizer.wss_std)
                acc["wss"] = acc["wss"] + wl / pg2
                acc["noslip"] = acc["noslip"] + noslip_loss(
                    self.networks, t["wx"], t["wy"], t["wz"], t["w_params"]) / pg2
                axial = int(g.meta.get("axial_dim", 0))
                acc["inlet"] = acc["inlet"] + inlet_velocity_loss(
                    self.networks, t["inlet_x"], t["inlet_y"], t["inlet_z"], t["inlet_params"],
                    float(g.meta.get("u_inlet_nd", 0.0)), axial_dim=axial) / pg2
                acc["outlet"] = acc["outlet"] + outlet_pressure_loss(
                    self.networks["p"], t["outlet_x"], t["outlet_y"], t["outlet_z"], t["outlet_params"])
                n_wall += 1
        if n_vel:
            acc["velocity"] /= n_vel
            acc["physics"] /= n_vel
        if n_wall:
            for c in ("pressure", "wss", "noslip", "inlet", "outlet"):
                acc[c] /= n_wall
        return acc

    def _weights(self, comp: Dict[str, torch.Tensor]) -> Dict[str, float]:
        aw = self.cfg.get("adaptive_weights", {})
        if not aw.get("enabled", False) or not self.adaptive_weights:
            w = self.cfg.get("loss_weights", {})
            return {c: float(w.get(c, 1.0)) for c in COMPONENTS}
        return dict(self.adaptive_weights)

    def _update_adaptive_weights(self, comp: Dict[str, torch.Tensor]) -> None:
        aw = self.cfg["adaptive_weights"]
        ref = aw.get("ref", "velocity")
        alpha = float(aw.get("alpha", 0.9))
        cap = float(aw.get("weight_cap", 20.0))
        floor = float(aw.get("physics_floor", 0.0))
        params = [p for n in self.networks.values() for p in n.parameters() if p.requires_grad]

        grad_norms = {}
        for c, loss in comp.items():
            if not loss.requires_grad or float(loss) == 0.0:
                continue
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            vals = [g.abs().mean().item() for g in grads if g is not None]
            grad_norms[c] = float(np.mean(vals)) if vals else 0.0
        if ref not in grad_norms:
            return
        ref_norm = grad_norms[ref]
        for c, gn in grad_norms.items():
            raw = ref_norm / max(gn, 1e-12)
            # EMA smoothing: alpha is the memory weight (Wang/Teng/Perdikaris 2021).
            # alpha=0.9 keeps 90% of the previous weight, damping the noisy
            # instantaneous gradient-norm ratio `raw`. Seeds to `raw` on the first
            # update for a component (get default), so no init discontinuity.
            self.adaptive_weights[c] = (
                alpha * self.adaptive_weights.get(c, raw) + (1 - alpha) * raw)
        for c in COMPONENTS:
            self.adaptive_weights.setdefault(c, 1.0)
            self.adaptive_weights[c] = min(self.adaptive_weights[c], cap)
        if floor > 0:
            self.adaptive_weights["physics"] = max(self.adaptive_weights["physics"], floor)

    # --------------------------------------------------------------- training
    def _physics_only(self) -> torch.Tensor:
        # Batched physics over all groups' collocation (C1 pg2 per group). Matches the
        # physics term of _component_losses; a fresh graph for the dedicated nut update.
        c = self._bz.get("coll")
        if c is None:
            return torch.zeros((), device=self.device)
        return self._physics_batched(c)

    def train_step(self) -> Dict[str, float]:
        aw = self.cfg.get("adaptive_weights", {})
        do_update = (aw.get("enabled", False) and self.epoch > 0
                     and self.epoch % int(aw.get("update_interval", 100)) == 0)

        self.opt.zero_grad(set_to_none=True)
        # Compute the component losses ONCE and reuse the single graph for both the
        # adaptive-weight update (per-component grad norms, retain_graph) and the
        # weighted backward. The old code built a SECOND _component_losses graph on
        # update epochs; with the batched (one big graph) losses the two graphs
        # coexisting spiked memory ~2x and OOM'd at every update_interval. Reusing
        # one graph is behaviorally identical (network weights are unchanged between
        # the two forwards, so the losses are the same) and also saves a forward.
        comp = self._component_losses()
        if do_update:
            self._update_adaptive_weights(comp)   # retain_graph -> graph survives below
        weights = self._weights(comp)
        total = sum(weights[c] * comp[c] for c in COMPONENTS)
        total.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for k, n in self.networks.items() if k != "nut" for p in n.parameters()],
                self.grad_clip)
        self.opt.step()

        # dedicated nut update on physics only
        self.opt_nut.zero_grad(set_to_none=True)
        ploss = self._physics_only()
        ploss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.networks["nut"].parameters(), self.grad_clip)
        self.opt_nut.step()

        return {"total": float(total), **{c: float(comp[c]) for c in COMPONENTS}}

    @torch.no_grad()
    def _velocity_rel_l2(self, data: Dict[str, torch.Tensor]) -> float:
        net_in = torch.cat([data["x"], data["y"], data["z"], data["params"]], dim=1)
        pred = torch.cat([self.networks["u"](net_in), self.networks["v"](net_in),
                          self.networks["w"](net_in)], dim=1)
        true = torch.cat([data["u_t"], data["v_t"], data["w_t"]], dim=1)
        return relative_l2(pred, true)

    def train(self) -> None:
        tr = self.cfg["training"]
        epochs = int(tr.get("epochs", 5000))
        eval_interval = int(tr.get("eval_interval", 500))
        save_interval = int(tr.get("save_interval", 1000))
        es = tr.get("early_stopping", {})
        patience = int(es.get("patience", 1000))
        min_delta = float(es.get("min_delta", 1e-6))
        es_enabled = bool(es.get("enabled", True))

        if self.start_epoch >= epochs:
            print(f"[trainer] resume target reached (start_epoch={self.start_epoch} "
                  f">= epochs={epochs}); nothing to train.")
            return
        where = f" (resumed from {self.start_epoch})" if self.start_epoch else ""
        print(f"[trainer] training {self.name} for {epochs} epochs on {self.device}{where}")
        t0 = time.time()
        for epoch in range(self.start_epoch + 1, epochs + 1):
            self.epoch = epoch
            losses = self.train_step()
            self.sched.step()
            self.sched_nut.step()

            # Stable model-selection monitor. The adaptive-weighted ``total`` is
            # non-stationary (its scale jumps when the loss weights update), so
            # it must NOT drive best-model/early-stop. Use the held-out velocity
            # rel-L2 when a holdout is configured (Stage A), else the unweighted
            # sum of raw component losses (stationary across weight updates).
            holdout_metric = self._velocity_rel_l2(self.holdout) if self.holdout is not None else None
            monitor = (holdout_metric if holdout_metric is not None
                       else sum(losses[c] for c in COMPONENTS))

            row = {"epoch": epoch, "lr": self.opt.param_groups[0]["lr"], "monitor": monitor, **losses}
            self.history.append(row)

            if epoch % eval_interval == 0 or epoch == 1:
                # Show ALL raw component losses (not just a subset) so training is
                # fully observable; loss_history.csv stores the same columns.
                comp = "  ".join(f"{c[:4]}={losses[c]:.2e}" for c in COMPONENTS)
                # Cumulative throughput + ETA: gives an honest wall-clock estimate
                # for the run (used for the CFD-vs-surrogate timing comparison).
                done = epoch - self.start_epoch
                eps = done / max(time.time() - t0, 1e-9)
                eta_min = (epochs - epoch) / eps / 60.0 if eps > 0 else float("nan")
                msg = (f"  ep {epoch:>6} lr={self.opt.param_groups[0]['lr']:.1e} "
                       f"total={losses['total']:.3e}  {comp}  monitor={monitor:.4f}"
                       f"  [{eps:.1f} ep/s, ETA {eta_min:.1f} min]")
                if holdout_metric is not None:
                    msg += f"  holdout_relL2={holdout_metric:.4f}"
                print(msg)

            # C2: a NaN/Inf monitor must NOT masquerade as success. `NaN < x` is
            # False in IEEE-754, so without this guard best_model.pt silently stops
            # updating and a diverged run finishes with a stale "best". Abort loudly.
            if not math.isfinite(monitor):
                print(f"[trainer] ABORT: non-finite monitor ({monitor}) at epoch {epoch} "
                      f"(diverged). Last finite best={self.best_loss:.4f}.")
                break
            improved = monitor < self.best_loss - min_delta
            if improved:
                self.best_loss = monitor
                self.no_improve = 0
                self.save_checkpoint("best_model.pt")
            else:
                self.no_improve += 1

            if epoch % save_interval == 0:
                self._save_history()
                self.save_resume_state()
            if es_enabled and self.no_improve >= patience:
                print(f"[trainer] early stop at epoch {epoch} (monitor={monitor:.4f})")
                break

        self._save_history()
        self.save_checkpoint("final_model.pt")
        # Clean finish -> drop the resume sidecar so a later --resume starts fresh.
        (self.out_dir / "resume_state.pt").unlink(missing_ok=True)
        mname = "holdout rel-L2" if self.holdout is not None else "unweighted loss sum"
        print(f"[trainer] done in {(time.time()-t0)/60:.1f} min. best {mname}={self.best_loss:.4f}")
        if self.holdout is not None:
            print(f"[trainer] final holdout velocity rel-L2 = {self._velocity_rel_l2(self.holdout):.4f}")

    # --------------------------------------------------------- crash recovery
    def save_resume_state(self) -> None:
        """Write full training state for crash recovery (separate from the clean
        inference checkpoints best_model.pt / final_model.pt). Written atomically
        via a temp file + replace so a kill mid-write cannot corrupt it."""
        state = {
            "epoch": self.epoch,
            "config": self.cfg,
            "networks": {k: n.state_dict() for k, n in self.networks.items()},
            "opt": self.opt.state_dict(),
            "opt_nut": self.opt_nut.state_dict(),
            "sched": self.sched.state_dict(),
            "sched_nut": self.sched_nut.state_dict(),
            "best_loss": self.best_loss,
            "no_improve": self.no_improve,
            "adaptive_weights": self.adaptive_weights,
            "history": self.history,
            "normalizer": self.normalizer.to_dict(),
            "Re": self._Re,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
                "numpy": np.random.get_state(),
            },
        }
        tmp = self.out_dir / "resume_state.pt.tmp"
        torch.save(state, tmp)
        tmp.replace(self.out_dir / "resume_state.pt")

    def _maybe_resume(self) -> None:
        if not bool(self.cfg.get("resume", False)):
            return
        path = self.out_dir / "resume_state.pt"
        if not path.exists():
            print(f"[trainer] resume requested but no {path.name}; starting fresh.")
            return
        state = torch.load(path, map_location=self.device, weights_only=False)
        # Guard against resurrecting an incompatible (e.g. historically leaky) run:
        # the sidecar carries the config it was written under. Refuse to continue if
        # the supervision sources or the training case list differ from the current
        # config, since that would restore weights/optimizer state trained on a
        # different data regime (e.g. XY+XZ plane-supervised) into a clean run.
        saved_data = (state.get("config") or {}).get("data", {})
        cur_data = self.cfg.get("data", {})
        saved_fp = (tuple(saved_data.get("velocity_kinds", ["3D"])),
                    tuple(saved_data.get("train_cases", [])))
        cur_fp = (tuple(cur_data.get("velocity_kinds", ["3D"])),
                  tuple(cur_data.get("train_cases", [])))
        if saved_fp != cur_fp:
            raise RuntimeError(
                f"refusing to resume from {path.name}: it was written with "
                f"velocity_kinds={saved_fp[0]} train_cases={saved_fp[1]}, but the "
                f"current config has velocity_kinds={cur_fp[0]} "
                f"train_cases={cur_fp[1]}. Delete the stale resume_state.pt or run "
                f"with a matching config.")
        for k, n in self.networks.items():
            n.load_state_dict(state["networks"][k])
        self.opt.load_state_dict(state["opt"])
        self.opt_nut.load_state_dict(state["opt_nut"])
        self.sched.load_state_dict(state["sched"])
        self.sched_nut.load_state_dict(state["sched_nut"])
        self.best_loss = float(state["best_loss"])
        self.no_improve = int(state["no_improve"])
        self.adaptive_weights = dict(state.get("adaptive_weights", {}))
        self.history = list(state.get("history", []))
        self.start_epoch = int(state["epoch"])
        rng = state.get("rng")
        if rng:
            try:
                torch.set_rng_state(rng["torch"])
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])
                np.random.set_state(rng["numpy"])
            except Exception as e:  # noqa: BLE001
                print(f"[trainer] RNG restore skipped: {e}")
        print(f"[trainer] RESUMED from epoch {self.start_epoch} "
              f"(best={self.best_loss:.4f}, no_improve={self.no_improve}, "
              f"{len(self.history)} history rows)")

    # ------------------------------------------------------------- checkpoint
    def save_checkpoint(self, filename: str) -> None:
        torch.save({
            "epoch": self.epoch,
            "config": self.cfg,
            "n_param": self.bundle.n_param,
            "networks": {k: n.state_dict() for k, n in self.networks.items()},
            "normalizer": self.normalizer.to_dict(),
            "Re": self._Re,
            "best_loss": self.best_loss,
            "adaptive_weights": self.adaptive_weights,
        }, self.out_dir / filename)

    def _save_history(self) -> None:
        if not self.history:
            return
        keys = list(self.history[0].keys())
        with open(self.out_dir / "loss_history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
        with open(self.out_dir / "normalizer.json", "w", encoding="utf-8") as f:
            json.dump(self.normalizer.to_dict(), f, indent=2)
