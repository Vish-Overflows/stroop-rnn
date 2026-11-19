"""Dynamical-systems analysis of trained networks (Sections 4.2 and 4.3 of the proposal).

For every network and every context (cue A on / cue B on, stimulus off) the analysis locates

1. the exact fixed points and the slow points reached by minimising q(h) = 1/2 ||F(h)||^2 (Eq. 2)
   from states sampled along trial trajectories;
2. the floor of the valley of q along the decision axis: for a grid of values of the decision
   variable d = y1 - y2 the slowest state with that value. If the network has an approximate line
   attractor these slow points are its backbone;
3. the linearised dynamics in firing-rate coordinates (Eq. 3) at the slow points around the decision
   boundary: the decision mode (eigenvalue closest to zero), the direction of the line attractor r
   and the selection vector l, averaged over these slow points;
4. the exact split of the context dependence of each feature's effective gain g = l . b into
   selection-vector modulation and input modulation (Eq. 5).

Three normalisations of the selection vector are reported. The primary one is that of Mante et al.
(2013): ||r|| = 1 and l . r = 1. In the second the selection vectors have unit norm, so only a
rotation of l between contexts counts as selection-vector modulation. In the third the mode is
scaled to the readout (w_d . r = 1), so that g is the rate of change of the decision variable per
unit input.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import FIXED_POINT_TOL, SLOW_EIG_TOL, TaskConfig
from .fixedpoints import (
    BELOW_TOL, RNNField, decision_mode, decompose_gain_change, effective_dimension, find_fixed_points,
    minimise_speed, rate_linearisation, slowest_point_with_readout, softplus, speed, static_gain_share,
    unique_points,
)
from .model import BatchedRNN, NoiseSource
from .task import TASK_A, TASK_B, make_sequences

CUES = {"neutral": (0.0, 0.0), "A": (1.0, 0.0), "B": (0.0, 1.0)}
VALLEY_D = (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)   # decision values of the slow points
CENTRE_D = 0.2            # slow points with |d| <= CENTRE_D enter the selection analysis
OP_CSI_MS = (100.0, 200.0, 400.0, 800.0)                        # operating points: states at stimulus onset
TRACK_OVERLAP = 0.8       # minimal |cos| between decision modes of neighbouring slow points


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().double().numpy()


def flow(field: RNNField, h0: np.ndarray, steps: int, dt: float) -> np.ndarray:
    """Noise-free Euler trajectory with the update used by the network simulation. [steps + 1, N]"""
    a = dt / field.tau
    out = np.empty((steps + 1, len(h0)))
    out[0] = h0
    for t in range(steps):
        out[t + 1] = out[t] + a * field.residual(out[t][None])[0]
    return out


def neutral_state(w_rec: np.ndarray, tau: float, dt: float, steps: int = 3000) -> dict:
    """The resting state without cue and stimulus: relax from h = 0, then converge to the fixed point."""
    field = RNNField(w_rec, np.zeros(w_rec.shape[0]), tau)
    h = flow(field, np.zeros(w_rec.shape[0]), steps, dt)[-1]
    res = minimise_speed(field, h[None], tol=1e-9)
    ev = np.linalg.eigvals(field.jacobian(res.h)[0])
    return {"h": res.h[0], "speed": float(res.speed[0]), "max_eig": float(ev.real.max()),
            "moved": float(np.linalg.norm(res.h[0] - h))}


def trace_valley(field: RNNField, w_d: np.ndarray, h_start: np.ndarray, d_values=VALLEY_D):
    """Slowest states with prescribed decision values, by continuation outwards from d = 0.

    Returns (points [P, N], speeds [P] in 1/ms).
    """
    d_values = np.asarray(d_values, dtype=float)
    if 0.0 not in d_values:
        raise ValueError("the decision values have to include 0")
    up = [i for i in np.argsort(d_values) if d_values[i] >= 0]
    down = [i for i in np.argsort(-d_values) if d_values[i] < 0]
    pts = np.empty((len(d_values), field.dim))
    spd = np.empty(len(d_values))
    for order in (up, down):
        h = h_start if order is up else pts[up[0]]
        for i in order:
            h, spd[i], _ = slowest_point_with_readout(field, w_d, d_values[i], h)
            pts[i] = h
    return pts, spd


def analyse_context(w_rec, bias, w_in, w_d, cfg: TaskConfig, h_neutral, ics, feature: int) -> dict:
    """Fixed points, slow points and the decision mode of one network in one context.

    feature is the index of the input that is relevant in this context.
    """
    field = RNNField(w_rec, bias, cfg.tau)
    tau = cfg.tau

    # operating points: noise-free states at stimulus onset for the four cue-stimulus intervals
    steps = [cfg.steps(ms) for ms in OP_CSI_MS]
    cue_traj = flow(field, h_neutral, max(steps), cfg.dt)
    h_op = cue_traj[steps]
    h_prep = h_op[-1]

    # slow points and exact fixed points reached from states along trial trajectories
    search = minimise_speed(field, ics, tol=FIXED_POINT_TOL)
    ok = search.status == BELOW_TOL
    slow = search.h[ok]
    slow = slow[unique_points(slow, 0.05)] if len(slow) else slow
    d_slow = softplus(slow) @ w_d if len(slow) else np.zeros(0)
    fps, _, _ = find_fixed_points(field, np.vstack([search.h, flow(field, h_prep, 3000, cfg.dt)[-1:]]), tol=1e-9)
    fp_unstable = np.array([int(np.sum(np.linalg.eigvals(field.jacobian(h[None])[0]).real > 1e-9)) for h in fps], dtype=int)
    fp_d = softplus(fps) @ w_d if len(fps) else np.zeros(0)

    # the valley of q along the decision axis
    d_values = np.asarray(VALLEY_D)
    pts, spd = trace_valley(field, w_d, h_prep, d_values)
    centre = int(np.flatnonzero(d_values == 0.0)[0])

    # decision mode at the decision boundary, then followed along the valley
    m0, gains0 = rate_linearisation(w_rec, pts[centre])
    ref = decision_mode(m0, w_d)
    b_ref = gains0[:, None] * w_in
    lams, overlaps, ls, bs, wr = [], [], [], [], []
    for h in pts:
        m, gains = rate_linearisation(w_rec, h)
        mode = decision_mode(m, w_d, r_ref=ref.r)
        lams.append(mode.lam)
        overlaps.append(mode.overlap)
        ls.append(mode.l)
        bs.append(gains[:, None] * w_in)
        wr.append(float(w_d @ mode.r))
    lams, overlaps, wr = np.array(lams), np.array(overlaps), np.array(wr)
    used = (np.abs(d_values) <= CENTRE_D + 1e-9) & (overlaps >= TRACK_OVERLAP)
    used[centre] = True
    l_mean = np.mean([ls[i] for i in np.flatnonzero(used)], axis=0)
    b_mean = np.mean([bs[i] for i in np.flatnonzero(used)], axis=0)
    wr_mean = float(np.mean(wr[used]))

    # the same linearisation at the prepared operating point (state at stimulus onset after the longest CSI)
    m_op, gains_op = rate_linearisation(w_rec, h_prep)
    mode_op = decision_mode(m_op, w_d)
    b_op = gains_op[:, None] * w_in

    inner = np.abs(d_values) <= CENTRE_D + 1e-9
    centred = pts - pts.mean(0)
    sv = np.linalg.svd(centred, compute_uv=False)
    return {
        # vectors
        "h_op": h_op, "valley": pts, "valley_speed": spd * tau, "valley_lam": lams, "valley_overlap": overlaps,
        "h_line": pts[centre], "l": l_mean, "b": b_mean, "wr": wr_mean,
        "l_ref": ref.l, "b_ref": b_ref, "r": ref.r, "wr_ref": float(w_d @ ref.r),
        "l_op": mode_op.l, "b_op": b_op, "r_op": mode_op.r, "wr_op": float(w_d @ mode_op.r),
        "eigvals_ref": ref.eigvals, "eigvals_op": mode_op.eigvals,
        "fixed_points": fps, "fp_unstable": fp_unstable, "fp_d": fp_d, "slow_points": slow,
        # scalars
        "lam": ref.lam, "lam_second": ref.second, "mode_is_slowest": ref.is_slowest,
        "slow_mode": bool(abs(ref.lam) < SLOW_EIG_TOL), "others_decay": bool(ref.second < -SLOW_EIG_TOL),
        "lam_mean": float(np.mean(lams[used])), "lam_max_abs": float(np.max(np.abs(lams[used]))),
        "n_points_averaged": int(used.sum()),
        "lam_op": mode_op.lam, "lam_second_op": mode_op.second, "op_mode_is_slowest": mode_op.is_slowest,
        "op_speed": float(speed(field, h_prep)[0] * tau),
        "op_to_line": float(np.linalg.norm(h_prep - pts[centre])),
        "cos_r_op": float(abs(mode_op.r @ ref.r)),
        "cond": ref.cond, "cos_lr": ref.cos_lr,
        "readout_overlap": float(abs(w_d @ ref.r) / np.linalg.norm(w_d)),
        "dc_share": static_gain_share(ref, w_d, b_ref[:, feature]),
        "speed_centre": float(spd[centre] * tau), "speed_max": float(spd[inner].max() * tau),
        "valley_length": float(np.linalg.norm(np.diff(pts[inner], axis=0), axis=1).sum()),
        "valley_pc1": float(sv[0] ** 2 / np.sum(sv**2)),
        "n_fixed": int(len(fps)), "n_stable": int(np.sum(fp_unstable == 0)),
        "fp_d_min": float(fp_d.min()) if len(fps) else float("nan"),
        "fp_d_max": float(fp_d.max()) if len(fps) else float("nan"),
        "n_slow": int(len(slow)), "slow_frac": float(np.mean(ok)),
        "slow_d_min": float(d_slow.min()) if len(slow) else float("nan"),
        "slow_d_max": float(d_slow.max()) if len(slow) else float("nan"),
    }


SCALAR_KEYS = (
    "lam", "lam_second", "mode_is_slowest", "slow_mode", "others_decay", "lam_mean", "lam_max_abs",
    "n_points_averaged", "lam_op", "lam_second_op", "op_mode_is_slowest", "op_speed", "op_to_line", "cos_r_op",
    "cond", "cos_lr", "readout_overlap", "dc_share", "speed_centre", "speed_max", "valley_length", "valley_pc1",
    "n_fixed", "n_stable", "fp_d_min", "fp_d_max", "n_slow", "slow_frac", "slow_d_min", "slow_d_max",
)

# linearisation points: (suffix, key of l, key of b, key of the readout projection w_d . r)
LINEARISATIONS = (("", "l", "b", "wr"), ("_ref", "l_ref", "b_ref", "wr_ref"), ("_op", "l_op", "b_op", "wr_op"))


def selection_summary(ctx: dict) -> dict:
    """Effective gains and the decomposition of Eq. (5) for both features of one network.

    ctx = {"A": analyse_context(...), "B": analyse_context(...)}. Keys without suffix use the selection
    vectors and input directions averaged over the slow points around the decision boundary; "_ref"
    uses the single slow point at d = 0 and "_op" the operating point at stimulus onset.
    """
    out = {}
    feature_context = {"A": ("A", "B"), "B": ("B", "A")}     # feature -> (relevant, irrelevant) context
    for k, feat in enumerate(("A", "B")):
        rel, irr = feature_context[feat]
        for tag, key_l, key_b, key_wr in LINEARISATIONS:
            l_rel, l_irr = ctx[rel][key_l], ctx[irr][key_l]
            b_rel, b_irr = ctx[rel][key_b][:, k], ctx[irr][key_b][:, k]
            dec = decompose_gain_change(l_rel, b_rel, l_irr, b_irr)
            for name, val in dec.items():
                out[f"{name}_{feat}{tag}"] = val
            # rotation only: unit-norm selection vectors
            unit = decompose_gain_change(l_rel / np.linalg.norm(l_rel), b_rel, l_irr / np.linalg.norm(l_irr), b_irr)
            # decision-variable units: l scaled by the readout projection of the line attractor
            dv = decompose_gain_change(l_rel * ctx[rel][key_wr], b_rel, l_irr * ctx[irr][key_wr], b_irr)
            for label, d in (("unit", unit), ("dv", dv)):
                for name in ("gain_rel", "gain_irr", "sel_mod", "inp_mod", "sel_fraction"):
                    out[f"{name}_{label}_{feat}{tag}"] = d[name]
            out[f"leak_{feat}{tag}"] = dv["gain_irr"] / dv["gain_rel"] if dv["gain_rel"] != 0 else float("nan")
            if tag == "":
                out[f"cos_b_{feat}"] = float(b_rel @ b_irr / (np.linalg.norm(b_rel) * np.linalg.norm(b_irr)))
                out[f"norm_b_rel_{feat}"] = float(np.linalg.norm(b_rel))
                out[f"norm_b_irr_{feat}"] = float(np.linalg.norm(b_irr))
    for tag, key_l, _, _ in LINEARISATIONS:
        l_a, l_b = ctx["A"][key_l], ctx["B"][key_l]
        out[f"cos_l{tag}"] = float(l_a @ l_b / (np.linalg.norm(l_a) * np.linalg.norm(l_b)))
        out[f"norm_l_A{tag}"] = float(np.linalg.norm(l_a))
        out[f"norm_l_B{tag}"] = float(np.linalg.norm(l_b))
    out["cos_r"] = float(ctx["A"]["r"] @ ctx["B"]["r"])
    return out


def analyse_network(w_rec, w_in, w_ctx, w_out, cfg: TaskConfig, ics: dict) -> tuple[dict, dict]:
    """All of the above for one network. ics = {"A": [K, N], "B": [K, N]} initial conditions of the search.

    Returns (row, detail): a flat dictionary of scalars and a dictionary with the vectors.
    """
    w_d = w_out[0] - w_out[1]
    neutral = neutral_state(w_rec, cfg.tau, cfg.dt)
    ctx = {}
    for k, cue in enumerate(("A", "B")):
        ctx[cue] = analyse_context(w_rec, w_ctx @ np.asarray(CUES[cue]), w_in, w_d, cfg, neutral["h"], ics[cue], k)
    row = {
        "edim": effective_dimension(w_rec),
        "norm_in_A": float(np.linalg.norm(w_in[:, 0])), "norm_in_B": float(np.linalg.norm(w_in[:, 1])),
        "norm_ctx_A": float(np.linalg.norm(w_ctx[:, 0])), "norm_ctx_B": float(np.linalg.norm(w_ctx[:, 1])),
        "cos_in": float(w_in[:, 0] @ w_in[:, 1] / (np.linalg.norm(w_in[:, 0]) * np.linalg.norm(w_in[:, 1]))),
        "neutral_speed": neutral["speed"] * cfg.tau, "neutral_max_eig": neutral["max_eig"],
    }
    for cue in ("A", "B"):
        for key in SCALAR_KEYS:
            row[f"{key}_ctx{cue}"] = ctx[cue][key]
    row.update(selection_summary(ctx))
    return row, {"h_neutral": neutral["h"], "ctx": ctx}


def trial_states(model: BatchedRNN, cfg: TaskConfig, context: int, h_neutral: torch.Tensor, n_trials: int,
                 seed: int, csi_ms: float = 400.0, hold_ms: float = 600.0) -> torch.Tensor:
    """States visited while the cue is on in noisy single trials of one context. [W, S, B, N]"""
    S = model.n_networks
    gen = torch.Generator().manual_seed(seed)
    dims = (S, n_trials, 1)
    seq = make_sequences(
        cfg, (S, n_trials), gen, n_trials=1,
        context=torch.full(dims, context), csi=torch.full(dims, cfg.steps(csi_ms)),
        rci=torch.full(dims, 1), hold=torch.full(dims, cfg.steps(hold_ms)), tail_steps=0,
    )
    h0 = h_neutral.view(S, 1, -1).expand(-1, n_trials, -1).float()
    _, h = model.simulate(seq.x, sigma_rec=cfg.sigma_rec, noise=NoiseSource(seed + 1), h0=h0, return_h=True)
    return h[1:]            # the cue is on from step 1


def analyse_checkpoint(model: BatchedRNN, cfg: TaskConfig, seed: int = 0, n_ic: int = 32, n_trials: int = 16):
    """Run the analysis for every network of a checkpoint; returns (rows, details), one entry per network."""
    S, N = model.n_networks, model.n_units
    w_rec, w_in, w_ctx, w_out = (_np(t) for t in (model.w_rec(), model.w_in, model.w_ctx, model.w_out))
    rng = np.random.default_rng(seed)
    h_neutral = np.stack([neutral_state(w_rec[s], cfg.tau, cfg.dt)["h"] for s in range(S)])
    h_neutral_t = torch.from_numpy(h_neutral).float()
    states = {cue: trial_states(model, cfg, context, h_neutral_t, n_trials, seed + 10 * context)
              for cue, context in (("A", TASK_A), ("B", TASK_B))}
    rows, details = [], []
    for s in range(S):
        ics = {}
        for cue in ("A", "B"):
            traj = states[cue][:, s]                                    # [W, B, N]
            pick = rng.integers(0, traj.shape[0] * traj.shape[1], size=n_ic)
            ics[cue] = _np(traj.reshape(-1, N)[pick])
        row, detail = analyse_network(w_rec[s], w_in[s], w_ctx[s], w_out[s], cfg, ics)
        rows.append(row)
        details.append(detail)
    return rows, details
