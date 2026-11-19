"""E0: validate the analysis pipeline on systems with known answers.

    (i)   hand-built linear line attractor: fixed points, direction r, selection vector l, effective gains
    (ii)  softplus network with a planted slow mode: fixed point, r, l, and the slow-point tracing
    (iii) Wong & Wang (2006): fixed points of the search against the intersections of the nullclines
    (iv)  rank-one networks (hypothesis V): the selection vector cannot rotate between contexts
    (v)   logistic regression: parameter recovery on simulated agents

    python scripts/e0_validation.py            # writes results/e0_validation.json and .npz

Part (iv) uses the trained rank-one networks and is skipped (with a note) if they are not available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from stroop_rnn.experiments import PSY_LEVELS
from stroop_rnn.fixedpoints import (
    GenericField, RNNField, decision_mode, decompose_gain_change, find_fixed_points, rate_linearisation,
    slowest_point_with_readout, softplus,
)
from stroop_rnn.stats import parameter_recovery
from stroop_rnn.validation import (
    WongWang, make_line_attractor, make_planted_rnn, rank_one_modes, simulate_linear,
    wong_wang_nullcline_fixed_points,
)

ROOT = Path(__file__).resolve().parents[1]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def linear_line_attractor(seeds=range(10), n: int = 20) -> tuple[dict, dict]:
    rows, arrays = [], {}
    for seed in seeds:
        sys = make_line_attractor(n, seed, lam=0.0)
        rng = np.random.default_rng(100 + seed)
        # fixed points from random initial conditions: they have to lie on the line span(r)
        field = GenericField(lambda x, m=sys.m: m @ x, n, tau=100.0)
        res_pts, _, _ = find_fixed_points(field, rng.normal(0, 2.0, size=(40, n)), tol=1e-10, merge_tol=1e-6)
        off_line = np.linalg.norm(res_pts - np.outer(res_pts @ sys.r, sys.r), axis=1)
        mode = decision_mode(sys.m)
        # Eq. (4): the displacement along the line after an input pulse is (l . b_k) * integral of u_k dt / tau
        u = np.zeros((300, 2))
        amp = rng.normal(size=2)
        u[:50] = amp
        x = simulate_linear(sys, np.zeros(n), u)
        moved = float(mode.l @ x[-1])
        predicted = float((mode.l @ sys.b) @ amp * 50 * 10.0 / 100.0)
        rows.append({
            "seed": int(seed), "n_fixed_points": int(len(res_pts)), "max_distance_from_line": float(off_line.max()),
            "spread_along_line": float(np.ptp(res_pts @ sys.r)),
            "eigenvalue_error": abs(mode.lam - sys.lam), "cos_r": _cos(mode.r, sys.r), "cos_l": _cos(mode.l, sys.l),
            "rel_error_l": float(min(np.linalg.norm(mode.l - sys.l), np.linalg.norm(mode.l + sys.l)) / np.linalg.norm(sys.l)),
            "gain_error": float(np.max(np.abs(np.abs(mode.l @ sys.b) - np.abs(sys.l @ sys.b)))),
            "cos_l_r": mode.cos_lr, "moved_along_line": moved, "predicted_from_gain": predicted,
        })
        if seed == 0:
            sign = np.sign(mode.l @ sys.l)
            arrays.update(lin_l_true=sys.l, lin_l_found=sign * mode.l, lin_r_true=sys.r, lin_r_found=sign * mode.r,
                          lin_traj_line=x @ mode.l, lin_traj_time=np.arange(len(x)) * 10.0)
    summary = {
        "n_systems": len(rows),
        "max_distance_from_line": max(r["max_distance_from_line"] for r in rows),
        "min_cos_r": min(r["cos_r"] for r in rows), "min_cos_l": min(r["cos_l"] for r in rows),
        "max_rel_error_l": max(r["rel_error_l"] for r in rows),
        "max_eigenvalue_error": max(r["eigenvalue_error"] for r in rows),
        "max_gain_error": max(r["gain_error"] for r in rows),
        "max_rel_error_displacement": max(abs(r["moved_along_line"] - r["predicted_from_gain"]) / abs(r["predicted_from_gain"]) for r in rows),
        "systems": rows,
    }
    arrays["lin_moved"] = np.array([r["moved_along_line"] for r in rows])
    arrays["lin_predicted"] = np.array([r["predicted_from_gain"] for r in rows])
    return summary, arrays


def planted_network(seeds=range(10), n: int = 40, lam: float = -0.02) -> tuple[dict, dict]:
    rows, arrays = [], {}
    for seed in seeds:
        net = make_planted_rnn(n, seed, lam=lam)
        rng = np.random.default_rng(200 + seed)
        field = RNNField(net.w_rec, net.bias, tau=100.0)
        pts, _, _ = find_fixed_points(field, net.h_star + rng.normal(0, 0.3, size=(30, n)), tol=1e-10)
        nearest = pts[np.argmin(np.linalg.norm(pts - net.h_star, axis=1))]
        m, gains = rate_linearisation(net.w_rec, nearest)
        mode = decision_mode(m)
        b = gains[:, None] * net.w_in
        # slow points traced along a random readout: near h* they have to move along the slow mode,
        # which is G^-1 r in the coordinates of h
        w = rng.normal(size=n) / np.sqrt(n)
        d0 = float(softplus(net.h_star) @ w)
        slow_dir = net.lin.r / expit(net.h_star)
        step = 0.02 * abs(float(w @ (expit(net.h_star) * slow_dir))) / np.linalg.norm(slow_dir)
        valley = _valley_around(field, w, net.h_star, d0, step)
        rows.append({
            "seed": int(seed), "n_fixed_points": int(len(pts)),
            "fixed_point_error": float(np.linalg.norm(nearest - net.h_star)),
            "eigenvalue_error": abs(mode.lam - lam), "cos_r": _cos(mode.r, net.lin.r), "cos_l": _cos(mode.l, net.lin.l),
            "gain_error": float(np.max(np.abs(np.abs(mode.l @ b) - np.abs(net.lin.l @ net.lin.b)))),
            "cos_valley_slow_direction": _cos(valley[2] - valley[0], slow_dir),
        })
    summary = {
        "n_networks": len(rows), "planted_eigenvalue": lam,
        "max_fixed_point_error": max(r["fixed_point_error"] for r in rows),
        "max_eigenvalue_error": max(r["eigenvalue_error"] for r in rows),
        "min_cos_r": min(r["cos_r"] for r in rows), "min_cos_l": min(r["cos_l"] for r in rows),
        "max_gain_error": max(r["gain_error"] for r in rows),
        "min_cos_valley_slow_direction": min(r["cos_valley_slow_direction"] for r in rows),
        "networks": rows,
    }
    return summary, arrays


def _valley_around(field: RNNField, w: np.ndarray, h_star: np.ndarray, d0: float, step: float) -> np.ndarray:
    """Slowest states with readout values d0 - step, d0, d0 + step (continuation from the fixed point)."""
    pts = [slowest_point_with_readout(field, w, d0 + s, h_star)[0] for s in (-step, 0.0, step)]
    return np.stack(pts)


def wong_wang() -> tuple[dict, dict]:
    cases = {"symmetric stimulus (mu0 = 30 Hz, c = 0)": dict(mu0=30.0, coh=0.0),
             "no stimulus (mu0 = 0)": dict(mu0=0.0, coh=0.0),
             "biased stimulus (mu0 = 30 Hz, c = 6.4%)": dict(mu0=30.0, coh=6.4)}
    out, arrays = {}, {}
    rng = np.random.default_rng(300)
    for i, (label, kw) in enumerate(cases.items()):
        model = WongWang(**kw)
        truth = wong_wang_nullcline_fixed_points(model)
        field = GenericField(lambda s, mdl=model: mdl.field(s) * mdl.tau_s, 2, tau=model.tau_s, eps=1e-7)
        found, _, counts = find_fixed_points(field, rng.uniform(0.0, 1.0, size=(200, 2)), tol=1e-9, merge_tol=1e-5)
        found = found[np.lexsort((found[:, 1], found[:, 0]))]
        truth = truth[np.lexsort((truth[:, 1], truth[:, 0]))]
        same_number = len(found) == len(truth)
        err = float(np.max(np.linalg.norm(found - truth, axis=1))) if same_number else float("nan")
        stable = [bool(np.all(np.linalg.eigvals(model.jacobian(p)).real < 0)) for p in found]
        out[label] = {"n_nullcline": int(len(truth)), "n_search": int(len(found)), "max_distance": err,
                      "n_stable": int(sum(stable)), "fixed_points": found.tolist(), "stable": stable}
        arrays[f"ww_found_{i}"], arrays[f"ww_truth_{i}"] = found, truth
        arrays[f"ww_stable_{i}"] = np.array(stable)
    return out, arrays


def rank_one(runs: Path) -> tuple[dict, dict]:
    # analytical check on random rank-one networks at random operating points ("contexts"); m and n
    # overlap, so that the non-trivial eigenvalue -1 + n^T G m / N is the slowest one
    rng = np.random.default_rng(400)
    errs = []
    for _ in range(20):
        n_units = 128
        m_vec = rng.normal(size=n_units)
        n_vec = 2.0 * m_vec + rng.normal(size=n_units)
        for _ctx in range(2):
            gains = expit(rng.normal(size=n_units))
            lam, r, l = rank_one_modes(m_vec, n_vec, gains)
            mat = -np.eye(n_units) + (gains * m_vec)[:, None] * n_vec[None, :] / n_units
            mode = decision_mode(mat)
            errs.append((abs(mode.lam - lam), 1 - _cos(mode.r, r), 1 - _cos(mode.l, l), 1 - _cos(mode.l, n_vec)))
    errs = np.array(errs)
    out = {"random_networks": {"n": 20, "max_eigenvalue_error": float(errs[:, 0].max()),
                               "max_1_minus_cos_r": float(errs[:, 1].max()), "max_1_minus_cos_l": float(errs[:, 2].max()),
                               "max_1_minus_cos_l_n": float(errs[:, 3].max())}}
    arrays = {}
    ckpt_path, an_path = runs / "rank1_k0_rho1.pt", runs / "analysis" / "rank1_k0_rho1.pt"
    if not (ckpt_path.exists() and an_path.exists()):
        out["trained_networks"] = "not available (train and analyse the rank-one networks first)"
        return out, arrays
    ckpt = torch.load(ckpt_path, weights_only=False)
    an = torch.load(an_path, weights_only=False)
    n_vecs = ckpt["state_dict"]["n"].double().numpy()
    dyn, bat = an["dynamics"], an["battery"]
    cos_ln = np.array([[_cos(det["ctx"][c]["l"], n_vecs[s]) for c in ("A", "B")] for s, det in enumerate(an["details"])])
    acc = np.minimum(bat["final_acc_A"], bat["final_acc_B"])
    # The sign of a selection vector follows the sign of its line attractor, which is chosen so that the
    # readout y1 - y2 increases along it. A rank-one network can read its single latent variable with
    # opposite signs in the two contexts; the two selection vectors then are the same axis with opposite
    # orientation (cosine -1). The "aligned" decomposition orients both selection vectors the same way.
    m_vecs = ckpt["state_dict"]["m"].double().numpy()
    w_out = ckpt["state_dict"]["w_out"].double().numpy()
    readout_sign = np.array([[np.sign((w_out[s, 0] - w_out[s, 1]) @ (expit(det["ctx"][c]["h_line"]) * m_vecs[s])) for c in ("A", "B")]
                             for s, det in enumerate(an["details"])])
    aligned = np.empty((len(acc), 2, 2))                    # [network, feature, (unit norm, Mante normalisation)]
    for s, det in enumerate(an["details"]):
        for k, (rel, irr) in enumerate((("A", "B"), ("B", "A"))):
            l_rel, l_irr = det["ctx"][rel]["l"], det["ctx"][irr]["l"]
            l_irr = l_irr * np.sign(l_rel @ l_irr)
            b_rel, b_irr = det["ctx"][rel]["b"][:, k], det["ctx"][irr]["b"][:, k]
            aligned[s, k, 0] = decompose_gain_change(l_rel / np.linalg.norm(l_rel), b_rel, l_irr / np.linalg.norm(l_irr), b_irr)["sel_fraction"]
            aligned[s, k, 1] = decompose_gain_change(l_rel, b_rel, l_irr, b_irr)["sel_fraction"]
    flipped = np.asarray(dyn["cos_l"]) < 0
    out["trained_networks"] = {
        "n": int(len(acc)), "min_accuracy": float(acc.min()), "n_above_90": int(np.sum(acc >= 0.9)),
        "min_cos_l_between_contexts": float(np.min(dyn["cos_l"])),
        "min_abs_cos_l_between_contexts": float(np.min(np.abs(dyn["cos_l"]))),
        "n_selection_vector_sign_flipped": int(flipped.sum()),
        "n_readout_sign_differs_between_contexts": int(np.sum(readout_sign[:, 0] != readout_sign[:, 1])),
        "sign_flip_explained_by_readout_sign": bool(np.array_equal(flipped, readout_sign[:, 0] != readout_sign[:, 1])),
        "sel_fraction_unit_aligned_max_abs": float(np.max(np.abs(aligned[..., 0]))),
        "sel_fraction_mante_aligned_max_abs": float(np.max(np.abs(aligned[..., 1]))),
        "sel_fraction_unit_max_abs_without_flipped": float(np.max(np.abs(
            np.stack([dyn["sel_fraction_unit_A"], dyn["sel_fraction_unit_B"]])[:, ~flipped]))) if (~flipped).any() else float("nan"),
        "min_cos_l_n": float(cos_ln.min()),
        "sel_fraction_mante_mean": float(np.mean([dyn["sel_fraction_A"], dyn["sel_fraction_B"]])),
        "sel_fraction_mante_max_abs": float(np.max(np.abs([dyn["sel_fraction_A"], dyn["sel_fraction_B"]]))),
        "sel_fraction_unit_max_abs": float(np.max(np.abs([dyn["sel_fraction_unit_A"], dyn["sel_fraction_unit_B"]]))),
        "sel_fraction_dv_max_abs": float(np.max(np.abs([dyn["sel_fraction_dv_A"], dyn["sel_fraction_dv_B"]]))),
        "decision_eigenvalue_mean": float(np.mean([dyn["lam_ctxA"], dyn["lam_ctxB"]])),
    }
    arrays.update(r1_cos_l=dyn["cos_l"], r1_cos_ln=cos_ln, r1_sel_fraction_aligned=aligned, r1_readout_sign=readout_sign,
                  r1_sel_fraction=np.stack([dyn["sel_fraction_A"], dyn["sel_fraction_B"]], axis=1),
                  r1_sel_fraction_unit=np.stack([dyn["sel_fraction_unit_A"], dyn["sel_fraction_unit_B"]], axis=1))
    return out, arrays


def logistic_recovery(n_rep: int = 40) -> tuple[dict, dict]:
    levels = np.asarray(PSY_LEVELS)
    a, b = np.meshgrid(levels, levels, indexing="ij")
    x = np.repeat(np.column_stack([a.ravel(), b.ravel()]), n_rep, axis=0)        # the design of E1 in one context
    rec = parameter_recovery(x, n_agents=200, seed=500, ranges=((-0.5, 0.5), (0.5, 8.0), (-2.0, 2.0)), ridge=1e-6)
    out = {"n_agents": 200, "n_trials": int(len(x)), "weights": ["bias", "feature 1", "feature 2"], **rec["summary"]}
    return out, {"rec_true": rec["true"], "rec_fitted": rec["fitted"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/main")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = {}, {}
    for name, func in (("linear_line_attractor", linear_line_attractor), ("planted_network", planted_network),
                       ("wong_wang", wong_wang), ("rank_one", lambda: rank_one(ROOT / args.runs)),
                       ("logistic_recovery", logistic_recovery)):
        report[name], arr = func()
        arrays.update(arr)
        print(f"--- {name}")
        print(json.dumps({k: v for k, v in report[name].items() if k not in ("systems", "networks")}, indent=2))
    (out_dir / "e0_validation.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out_dir / "e0_validation.npz", **arrays)
    print(f"written {out_dir / 'e0_validation.json'}")


if __name__ == "__main__":
    main()
