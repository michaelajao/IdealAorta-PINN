"""Fit one neural-field arm to one reconstruction problem and score it.

Arms share the network (``fields.FieldNet``), optimizer, schedule, batch sizes and
observations; they differ only in the physics added to the loss:

    data        velocity observations + wall no-slip + inlet plug + outlet gauge
    cont        data + continuity
    steady      cont + steady RANS momentum
    unsteady    cont + short-window unsteady RANS momentum

``closure`` sets the viscosity inside the momentum residual: ``oracle`` uses the CFD
eddy viscosity at the collocation nodes (not deployable; an information arm), ``lam``
the molecular value only. The physics weight is chosen per arm on the shared 10 %
observation hold-out (``problem.validation_split``) and then frozen.

Every run is scored twice: through the shared pipeline (``postprocess.score_field``,
identical to the interpolation comparators) and with its own heads (autograd wall
shear, pressure network).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import CHECKS_DIR, MODELS_DIR, MU, RHO, RUNS_DIR
from .fields import FieldNet, residuals_func, wall_shear_s
from .postprocess import PressureIntegrator, pressure_scores, score_field, wss_scores
from .problem import build_problem, validation_split

ARMS = ("data", "cont", "steady", "unsteady")
CLOSURES = ("oracle", "lam")
DEFAULT_STEPS = 20000
BATCH = {"obs": 4096, "col": 4096, "wall": 2048, "in": 512, "out": 256}
_SOURCES = [Path(__file__).parent / f for f in ("fields.py", "problem.py", "postprocess.py", "training.py")]


@dataclass
class ArmSpec:
    arm: str
    case: int
    times: List[int]
    target: int
    closure: str = "oracle"
    grid: float = 2.5
    seed: int = 0              # observation-grid draw
    init_seed: int = 0         # network / optimizer seed
    wphys: float = 0.01
    steps: int = DEFAULT_STEPS
    lr: float = 1e-3
    min_wall_mm: float = 0.0   # drop observations nearer the wall than this

    @property
    def name(self) -> str:
        """Run name; a step count other than the default is part of the name, so a short
        selection run and a full run of the same arm never share a record."""
        steps = "" if self.steps == DEFAULT_STEPS else f"_n{self.steps}"
        return (f"{self.arm}_{self.closure}_c{self.case:02d}_t{self.target}_g{self.grid:g}"
                f"_s{self.seed}_i{self.init_seed}_w{self.wphys:g}{steps}"
                + (f"_mw{self.min_wall_mm:g}" if self.min_wall_mm > 0 else ""))


def source_hash() -> str:
    h = hashlib.sha256()
    for p in _SOURCES:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def train_arm(spec: ArmSpec, device: str = "cuda", profile_steps: int = 0, verbose: bool = True) -> Dict:
    """Train one arm, score it, save ``models/<name>/model.pt`` and the run record.

    With ``profile_steps`` > 0 only that many steps are timed and a cost estimate is returned.
    """
    assert spec.arm in ARMS and spec.closure in CLOSURES, spec
    torch.manual_seed(spec.init_seed)
    dev = torch.device(device)
    t_start = time.time()

    P = build_problem(spec.case, spec.times, spec.target, spec.grid, spec.seed, min_wall_mm=spec.min_wall_mm)
    t_lo, t_hi = sorted(spec.times)
    tc, tau = 0.5 * (t_lo + t_hi) / 1000, 0.5 * (t_hi - t_lo) / 1000
    xc = P.X.mean(0)
    L = 0.5 * float((P.X.max(0) - P.X.min(0)).max())
    val_mask = validation_split(P.obs_ids, spec.init_seed)
    obs_tr, obs_val = P.obs_ids[~val_mask], P.obs_ids[val_mask]
    U = float(np.quantile(np.linalg.norm(P.snaps[spec.target].uvw[obs_tr], axis=1), 0.995))
    time_coef = L / (U * tau)

    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    xs_all = T((P.X - xc) / L)
    tin = {t: (t / 1000 - tc) / tau for t in spec.times}
    ob_x = torch.cat([xs_all[obs_tr]] * 2)
    ob_t = torch.cat([torch.full((len(obs_tr), 1), tin[t], device=dev) for t in spec.times])
    ob_u = torch.cat([T(P.snaps[t].uvw[obs_tr] / U) for t in spec.times])
    wall_x = T((P.wall_xyz - xc) / L)
    in_x = T((P.inlet_xyz - xc) / L)
    in_u = {t: T(P.snaps[t].inlet_uvw / U) for t in spec.times}
    out_x = T((P.outlet_xyz - xc) / L)
    nu_mol = MU / (RHO * U * L)
    mut = {t: T(P.snaps[t].mut / (RHO * U * L)) for t in spec.times}
    gmut = {t: T(P.snaps[t].grad_mut / (RHO * U)) for t in spec.times}   # d nu_s / d x_s

    net = FieldNet(seed=spec.init_seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=spec.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, spec.steps, eta_min=spec.lr * 1e-2)
    use_mom = spec.arm in ("steady", "unsteady")
    use_cont = spec.arm != "data"
    rand_t = lambda n: torch.rand(n, 1, device=dev) * 2 - 1  # noqa: E731
    history, step_times = [], []

    for step in range(spec.steps):
        ts = time.time()
        i = torch.randint(len(ob_x), (BATCH["obs"],), device=dev)
        l_data = ((net(ob_x[i], ob_t[i])[:, :3] - ob_u[i]) ** 2).sum(1).mean()
        iw = torch.randint(len(wall_x), (BATCH["wall"],), device=dev)
        l_wall = (net(wall_x[iw], rand_t(BATCH["wall"]))[:, :3] ** 2).sum(1).mean()
        l_in, l_gauge = 0.0, 0.0
        for t in spec.times:
            ii = torch.randint(len(in_x), (BATCH["in"],), device=dev)
            t_in = torch.full((BATCH["in"], 1), tin[t], device=dev)
            l_in = l_in + ((net(in_x[ii], t_in)[:, :3] - in_u[t][ii]) ** 2).sum(1).mean()
            io = torch.randint(len(out_x), (BATCH["out"],), device=dev)
            l_gauge = l_gauge + net(out_x[io], torch.full((BATCH["out"], 1), tin[t], device=dev))[:, 3].mean() ** 2
        loss = l_data + l_wall + l_in + l_gauge
        l_c = l_m = torch.zeros((), device=dev)
        if use_cont:
            ic = torch.randint(len(xs_all), (BATCH["col"],), device=dev)
            xcol = xs_all[ic].clone().requires_grad_(True)
            tcol = rand_t(BATCH["col"]).requires_grad_(True)
            a = (tcol + 1) / 2                                   # linear-in-time closure between snapshots
            if spec.closure == "oracle":
                nu = nu_mol + (1 - a) * mut[t_lo][ic, None] + a * mut[t_hi][ic, None]
                gnu = (1 - a) * gmut[t_lo][ic] + a * gmut[t_hi][ic]
            else:
                nu = torch.full_like(a, nu_mol)
                gnu = torch.zeros(BATCH["col"], 3, device=dev)
            r = residuals_func(net, xcol, tcol, nu, gnu, time_coef, steady=(spec.arm == "steady"), momentum=use_mom)
            l_c = (r["cont"] ** 2).mean()
            loss = loss + spec.wphys * l_c
            if use_mom:
                l_m = (r["mom"] ** 2).sum(1).mean()
                loss = loss + spec.wphys * l_m
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if profile_steps:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.time() - ts)
            if step + 1 >= profile_steps:
                s_step = float(np.median(step_times[5:]))
                return {"arm": spec.arm, "s_per_step_median": s_step,
                        "peak_GB": torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0,
                        "est_hours_for_steps": s_step * spec.steps / 3600}
        if step % 500 == 0 or step == spec.steps - 1:
            with torch.no_grad():
                vx = torch.cat([xs_all[obs_val]] * 2)
                vt = torch.cat([torch.full((len(obs_val), 1), tin[t], device=dev) for t in spec.times])
                vu = torch.cat([T(P.snaps[t].uvw[obs_val] / U) for t in spec.times])
                val = float(torch.linalg.norm(net(vx, vt)[:, :3] - vu) / torch.linalg.norm(vu))
            history.append({"step": step, "loss": float(loss), "data": float(l_data), "wall": float(l_wall),
                            "cont": float(l_c), "mom": float(l_m), "val_obs_rel_l2": val})
            if verbose:
                print(history[-1], flush=True)

    # ---------------- scoring on the hidden targets ----------------
    net.eval()
    scales = {"xc": xc, "L": L, "U": U, "tc": tc, "tau": tau}
    fields = {t: predict(net, P.X, t, scales, dev) for t in spec.times}
    integ = PressureIntegrator(P.X, P.wall_xyz, P.dwall)
    record = {"method": "nf", "run_name": spec.name, "spec": asdict(spec), "problem": P.info(),
              "scales": {"U": U, "L": L, "tau_s": tau},
              "n_obs_train_per_time": int(len(obs_tr)), "n_obs_val_per_time": int(len(obs_val)),
              "source_sha": source_hash(), "history": history,
              "final_val_obs_rel_l2": history[-1]["val_obs_rel_l2"],
              **score_field(P, integ, {t: f[:, :3] for t, f in fields.items()})}
    tgt = P.snaps[spec.target]
    if P.wall_aneurysm.any():
        W = wall_shear_physical(net, P.wall_xyz, P.wall_normals, spec.target, scales, dev)
        record["wss_autograd"] = wss_scores(W, tgt.wall_wss, P.wall_aneurysm)
    p_own = fields[spec.target][:, 3][integ.sub]
    record["pressure_own_head"] = pressure_scores(p_own - p_own[integ.o_idx].mean(), integ.gauge_ref(tgt.p),
                                                  integ, P.sac[integ.sub])
    record["seconds"] = round(time.time() - t_start, 1)
    save_run(record)
    ck = MODELS_DIR / spec.name
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), **scales, "spec": asdict(spec)}, ck / "model.pt")
    return record


def predict(net: FieldNet, X: np.ndarray, t_ms: int, scales: Dict, device, chunk: int = 65536) -> np.ndarray:
    """Physical (u, v, w, p_gauge) at nodes ``X`` and time ``t_ms``."""
    out = []
    t_in = (t_ms / 1000 - scales["tc"]) / scales["tau"]
    with torch.no_grad():
        for k in range(0, len(X), chunk):
            xb = torch.as_tensor((X[k:k + chunk] - scales["xc"]) / scales["L"], dtype=torch.float32, device=device)
            out.append(net(xb, torch.full((len(xb), 1), t_in, device=device)).cpu().numpy())
    f = np.concatenate(out).astype(np.float64)
    f[:, :3] *= scales["U"]
    f[:, 3] *= RHO * scales["U"] ** 2
    return f


def wall_shear_physical(net: FieldNet, wall_xyz: np.ndarray, normals: np.ndarray, t_ms: int, scales: Dict,
                        device, chunk: int = 16384) -> np.ndarray:
    """Newtonian wall shear (Pa) from the network's own velocity gradient."""
    W = np.zeros((len(wall_xyz), 3))
    t_in = (t_ms / 1000 - scales["tc"]) / scales["tau"]
    for k in range(0, len(wall_xyz), chunk):
        xb = torch.as_tensor((wall_xyz[k:k + chunk] - scales["xc"]) / scales["L"], dtype=torch.float32,
                             device=device).requires_grad_(True)
        tb = torch.full((len(xb), 1), t_in, device=device).requires_grad_(True)
        nb = torch.as_tensor(normals[k:k + chunk], dtype=torch.float32, device=device)
        W[k:k + chunk] = wall_shear_s(net, xb, tb, nb).detach().cpu().numpy() * MU * scales["U"] / scales["L"]
    return W


def load_field(name: str, device: str = "cpu"):
    """Reload a trained arm: ``(net, scales, spec)``."""
    ck = torch.load(MODELS_DIR / name / "model.pt", map_location=device, weights_only=False)
    spec = ck.get("spec") or ck["args"]           # older checkpoints stored the CLI args
    net = FieldNet(seed=spec["init_seed"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, {k: ck[k] for k in ("xc", "L", "U", "tc", "tau")}, spec


def save_run(record: Dict, name: Optional[str] = None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name or record['run_name']}.json"
    path.write_text(json.dumps(record, indent=1, default=float))
    return path


DIAG_BINS_M = (0, 0.25e-3, 0.5e-3, 1e-3, 2e-3, 5e-3, 1.0)


def diagnose(name: str, device: str = "cuda") -> Dict:
    """Where a trained field's momentum residual sits, and its pipeline pressure.

    Reports, at the target time, the unsteady + oracle-mu_t momentum and continuity
    residuals binned by wall distance with the fraction of collocation nodes per bin
    (nodes are sampled uniformly, so the refined prism layers are over-represented),
    and the pressure integrated from the field's velocity in every pressure form.
    Diagnostic only: nothing here feeds back into a scored run.
    """
    net, scales, spec = load_field(name, device)
    P = build_problem(spec["case"], spec["times"], spec["target"], spec["grid"], spec["seed"], save_mask=False,
                      min_wall_mm=spec.get("min_wall_mm", 0.0))
    L, U, tau = scales["L"], scales["U"], scales["tau"]
    T = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)  # noqa: E731
    t_in = (spec["target"] / 1000 - scales["tc"]) / tau
    tgt = P.snaps[spec["target"]]
    rng = np.random.default_rng(0)
    out = {"name": name, "bins_m": list(DIAG_BINS_M), "residual": []}
    for lo, hi in zip(DIAG_BINS_M[:-1], DIAG_BINS_M[1:]):
        m = np.flatnonzero((P.dwall >= lo) & (P.dwall < hi))
        idx = rng.choice(m, min(4000, len(m)), replace=False)
        xs = T((P.X[idx] - scales["xc"]) / L).requires_grad_(True)
        tt = torch.full((len(idx), 1), t_in, device=device).requires_grad_(True)
        nu = T((MU + tgt.mut[idx]) / (RHO * U * L))[:, None]
        gnu = T(tgt.grad_mut[idx] / (RHO * U))
        r = residuals_func(net, xs, tt, nu, gnu, L / (U * tau), steady=False)
        mom = r["mom"].detach().cpu().numpy()
        out["residual"].append({"dwall_mm": [lo * 1e3, hi * 1e3], "node_fraction": len(m) / len(P.X),
                                "mom_rms": float(np.sqrt((mom ** 2).sum(1).mean())),
                                "mom_median": float(np.median(np.linalg.norm(mom, axis=1))),
                                "cont_rms": float(r["cont"].detach().pow(2).mean().sqrt())})
    fields = {t: predict(net, P.X, t, scales, device)[:, :3] for t in spec["times"]}
    integ = PressureIntegrator(P.X, P.wall_xyz, P.dwall)
    scored = score_field(P, integ, fields)
    out["pipeline"] = scored["pressure"]
    out["pipeline_wss"] = scored.get("wss")
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    (CHECKS_DIR / f"diag_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    return out
