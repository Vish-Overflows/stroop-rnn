"""Collect the per-condition analysis files into the tables used for statistics and figures.

    python scripts/collect_results.py           # runs/main/analysis -> results/

Writes
    results/networks.csv           one row per network: condition, accuracy, behavioural and dynamical measures
    results/arrays.npz             per-network arrays (same order as networks.csv): cell means, psychometric
                                   curves, perturbations, state resets, slow points, eigenvalue spectra
    results/snapshots.csv          behaviour of the networks at snapshots during joint training
    results/training_history.npz   loss and accuracy during training, per condition
    results/example_network.npz    state trajectories of one baseline network (for the state-space figure)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stroop_rnn import experiments as ex
from stroop_rnn.config import MAIN_THRESHOLD, MIN_ACCURACY, TrainConfig, task_config_from
from stroop_rnn.model import BatchedRNN, NoiseSource
from stroop_rnn.task import make_sequences

ROOT = Path(__file__).resolve().parents[1]
CONDITION_KEYS = ("variant", "k", "rho", "norm_control", "optimizer")


def _scalars(prefix: str, d: dict, n: int) -> dict:
    """All entries of d that are one number per network."""
    out = {}
    for key, val in d.items():
        arr = np.asarray(val)
        if arr.shape == (n,) and arr.dtype.kind in "fiub":
            out[f"{prefix}{key}"] = arr.astype(float)
    return out


def network_table(an: dict) -> pd.DataFrame:
    cond = an["condition"]
    n = len(an["seeds"])
    cols: dict = {"condition": [cond["name"]] * n, "net": np.arange(n), "seed": np.asarray(an["seeds"])}
    for key in CONDITION_KEYS:
        cols[key] = [cond[key]] * n
    cols["train_steps"] = [an["train_steps"]] * n
    cols["train_loss"] = np.asarray(an["history"]["loss"][-1], dtype=float)
    bat = an["battery"]
    cols.update(_scalars("", bat, n))
    cols["min_acc"] = np.minimum(bat["final_acc_A"], bat["final_acc_B"])
    cols["included"] = cols["min_acc"] >= MIN_ACCURACY
    if "dynamics" in an:
        dyn = an["dynamics"]
        cols.update(_scalars("", dyn, n))
        # intrusion of the irrelevant feature relative to the relevant feature of the same context
        cols["intrusion_on_A"] = dyn["gain_irr_dv_B"] / dyn["gain_rel_dv_A"]      # feature B while task A is cued
        cols["intrusion_on_B"] = dyn["gain_irr_dv_A"] / dyn["gain_rel_dv_B"]      # feature A while task B is cued
        beta = an["psychometric"]["beta"]                                          # [n, context, (1, u_A, u_B)]
        for c, cue in enumerate("AB"):
            for j, name in enumerate(("b0", "bA", "bB")):
                cols[f"psy_{name}_ctx{cue}"] = beta[:, c, j]
        cols["psy_intrusion_on_A"] = beta[:, 0, 2] / beta[:, 0, 1]
        cols["psy_intrusion_on_B"] = beta[:, 1, 1] / beta[:, 1, 2]
        pert = an["perturbations"]
        amps = pert["amplitudes"]
        sens = (pert["dd_free"][..., 0] - pert["dd_free"][..., 1]) / (2 * amps)   # [n, ctx, D, A]
        bias = pert["p_plus"][..., 0] - pert["p_plus"][..., 1]
        for name, sl in (("sel", slice(0, 1)), ("line", slice(1, 2)), ("rand", slice(2, None))):
            cols[f"pert_sens_{name}"] = sens[:, :, sl, 0].mean(axis=(1, 2))
            cols[f"pert_pred_{name}"] = pert["dd_pred"][:, :, sl].mean(axis=(1, 2))
            for ai, amp in enumerate(amps):
                cols[f"pert_bias_{name}_a{amp:g}"] = bias[:, :, sl, ai].mean(axis=(1, 2))
        rel = an["relaxation"]
        for si, start in enumerate(rel["starts"]):
            for c, cue in enumerate("AB"):
                for key in ("dist0", "t_half", "t_e", "tau_tail"):
                    cols[f"relax_{key}_{start}_{cue}"] = rel[key][:, c, si]
        for c, cue in enumerate("AB"):
            cols[f"relax_tau_eig2_{cue}"] = rel["tau_eig2"][:, c]
        if "step" in an:
            cols.update(step_columns(an))
    return pd.DataFrame(cols)


STEP_TIMES_MS = (50, 100, 200, 500)


def step_columns(an: dict) -> dict:
    """Noise-free responses to the two features: one number per network.

    nf_rel_<task>_csi<ms>_t<ms>, nf_irr_...   response of the decision variable to the relevant / irrelevant
                                               feature alone, t ms after stimulus onset
    nf_rt_<congruency>_<task>_csi<ms>         noise-free response time; nf_interference_*, nf_facilitation_*
    nf_slope_rel_<task>, nf_slope_irr_<task>  predicted initial slope of the decision variable (per ms)
    nf_slope_meas_rel_<task>, ..._irr_<task>  measured change over the first time step (per ms), longest CSI
    """
    cfg = task_config_from(an["task_config"])
    step = an["step"]
    sm = ex.step_summary(step, cfg, MAIN_THRESHOLD)
    cols = {}
    for ti, task in enumerate("AB"):
        cols[f"nf_slope_rel_{task}"] = sm["slope_rel"][:, ti]
        cols[f"nf_slope_irr_{task}"] = sm["slope_irr"][:, ti]
        cols[f"nf_slope_meas_rel_{task}"] = sm["resp_rel"][:, ti, -1, 0] / cfg.dt
        cols[f"nf_slope_meas_irr_{task}"] = sm["resp_irr"][:, ti, -1, 0] / cfg.dt
        for ci, ms in enumerate(step["csi_ms"]):
            tag = f"{task}_csi{int(ms)}"
            for j, cong in enumerate(("cong", "neut", "incong")):
                cols[f"nf_rt_{cong}_{tag}"] = sm["rt"][:, ti, ci, j]
            cols[f"nf_interference_{tag}"] = sm["interference"][:, ti, ci]
            cols[f"nf_facilitation_{tag}"] = sm["facilitation"][:, ti, ci]
            for t_ms in STEP_TIMES_MS:
                idx = cfg.steps(t_ms) - 1
                cols[f"nf_rel_{tag}_t{t_ms}"] = sm["resp_rel"][:, ti, ci, idx]
                cols[f"nf_irr_{tag}_t{t_ms}"] = sm["resp_irr"][:, ti, ci, idx]
        if "default" in sm:                                   # response to the feature when no task is cued
            for t_ms in STEP_TIMES_MS:
                cols[f"nf_default_{task}_t{t_ms}"] = sm["default"][:, ti, cfg.steps(t_ms) - 1]
    return cols


def network_arrays(an: dict) -> dict:
    bat = an["battery"]
    out = {k: bat[k] for k in ("cell_rt", "cell_acc", "cell_premature", "cell_n", "cong_rt", "cong_acc", "stroop_rt", "stroop_acc")}
    if "dynamics" not in an:
        return out
    psy, pert, e5, lr, rel = (an[k] for k in ("psychometric", "perturbations", "interventions", "long_rsi", "relaxation"))
    out.update(psy_p_plus=psy["p_plus"], psy_beta=psy["beta"], psy_se=psy["se"])
    out.update(pert_dd_free=pert["dd_free"], pert_p_plus=pert["p_plus"], pert_p_plus0=pert["p_plus0"],
               pert_dd_pred=pert["dd_pred"])
    for key in ("rt", "acc", "premature", "rt_raw"):
        out[f"e5_{key}"] = e5[key]
        out[f"long_{key}"] = lr[key]
    out["relax_dist_csi"] = rel["dist_csi"]
    if "dist" in rel:
        out["relax_dist"] = rel["dist"]                       # [n, context, start, time]
    if "step" in an:
        sm = ex.step_summary(an["step"], task_config_from(an["task_config"]), MAIN_THRESHOLD)
        out.update(step_resp_rel=sm["resp_rel"], step_resp_irr=sm["resp_irr"], step_rt=sm["rt"])
        if "default" in sm:
            out["step_default"] = sm["default"]               # [n, feature, time]: response without any cue
    det = an["details"]
    for key in ("valley_speed", "valley_lam", "valley_overlap", "eigvals_ref", "eigvals_op"):
        out[key] = np.stack([np.stack([d["ctx"][c][key] for c in "AB"]) for d in det])
    out["fp_d"] = np.stack([np.stack([_pad(d["ctx"][c]["fp_d"], 8) for c in "AB"]) for d in det])
    out["fp_unstable"] = np.stack([np.stack([_pad(d["ctx"][c]["fp_unstable"], 8) for c in "AB"]) for d in det])
    return out


def _pad(values, size: int) -> np.ndarray:
    """A fixed-length copy of a short list (NaN-padded), so that per-network lists can be stacked."""
    out = np.full(size, np.nan)
    values = np.asarray(values, dtype=float)[:size]
    out[:len(values)] = values
    return out


def example_network(runs: Path, condition: str = "full_k0_rho1", net: int = 0, n_trials: int = 6, seed: int = 99) -> dict:
    """Noise-averaged trajectories of one network in the plane of the line attractor and the selection vectors."""
    ckpt = torch.load(runs / f"{condition}.pt", weights_only=False)
    an = torch.load(runs / "analysis" / f"{condition}.pt", weights_only=False)
    model = BatchedRNN.from_checkpoint(ckpt)
    cfg = task_config_from(ckpt["task_config"])
    det = an["details"][net]
    S, N = model.n_networks, model.n_units
    h0 = torch.from_numpy(np.stack([d["h_neutral"] for d in an["details"]])).float()
    out = {"net": net, "condition": condition}
    w_out = model.w_out.detach().double().numpy()[net]
    w_d = w_out[0] - w_out[1]
    levels = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)
    for c, cue in enumerate("AB"):
        ctx = det["ctx"][cue]
        # every combination of signed strengths of both features, averaged over noise
        vals = torch.tensor([(a, b) for a in levels for b in levels])
        B = len(vals) * n_trials
        va, vb = vals[:, 0].repeat(n_trials), vals[:, 1].repeat(n_trials)
        context = torch.full((S, B, 1), c)
        rel, irr = (va, vb) if c == 0 else (vb, va)
        seq = make_sequences(cfg, (S, B), torch.Generator().manual_seed(seed + c), n_trials=1, context=context,
                             rel=rel.view(1, B, 1).expand(S, -1, -1) * cfg.signal, irr=irr.view(1, B, 1).expand(S, -1, -1) * cfg.signal,
                             csi=torch.full((S, B, 1), cfg.steps(800.0)), rci=torch.full((S, B, 1), 1),
                             hold=torch.full((S, B, 1), cfg.steps(400.0)), tail_steps=0)
        _, h = model.simulate(seq.x, sigma_rec=cfg.sigma_rec, noise=NoiseSource(seed + 10 + c),
                              h0=h0.view(S, 1, N).expand(-1, B, -1), return_h=True)
        rates = torch.nn.functional.softplus(h[:, net]).double().numpy()                  # [T, B, N]
        rates = rates.reshape(rates.shape[0], n_trials, len(vals), N).mean(axis=1)          # [T, 36, N]
        centre = np.log1p(np.exp(ctx["h_line"]))
        dr = rates - centre
        out[f"proj_line_{cue}"] = dr @ ctx["l"]                 # position along the line attractor (l . dr)
        out[f"proj_d_{cue}"] = rates @ w_d                      # decision variable
        # input axes: the part of each feature's input direction that is orthogonal to the line attractor
        for k, feat in enumerate("AB"):
            b = ctx["b"][:, k]
            b_perp = b - (b @ ctx["r"]) * ctx["r"]
            out[f"proj_in{feat}_{cue}"] = dr @ (b_perp / np.linalg.norm(b_perp))
        out[f"valley_line_{cue}"] = (np.log1p(np.exp(ctx["valley"])) - centre) @ ctx["l"]
        out[f"valley_d_{cue}"] = np.log1p(np.exp(ctx["valley"])) @ w_d
        out[f"values_{cue}"] = vals.numpy()
        out[f"stim_on_{cue}"] = int(seq.stim_on[0, 0, 0])
        out[f"stim_off_{cue}"] = int(seq.stim_off[0, 0, 0])

    # one back-to-back test sequence with the output of the network (for the task figure)
    seq = make_sequences(cfg, (S, 1), torch.Generator().manual_seed(seed + 100))
    y = model.simulate(seq.x, sigma_rec=cfg.sigma_rec, noise=NoiseSource(seed + 101))
    out["seq_x"] = seq.x[:, net, 0].numpy()
    out["seq_target"] = seq.target[:, net, 0].numpy()
    out["seq_y"] = y[:, net, 0].detach().numpy()
    for key in ("context", "congruency", "switch", "cue_on", "stim_on", "stim_off", "cue_off"):
        out[f"seq_{key}"] = getattr(seq, key)[net, 0].numpy()
    out["dt"] = cfg.dt
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/main")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    runs, out_dir = ROOT / args.runs, ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tables, arrays, history = [], {}, {}
    files = sorted((runs / "analysis").glob("*.pt"))
    if not files:
        raise SystemExit(f"no analysis files in {runs / 'analysis'}")
    for path in files:
        an = torch.load(path, weights_only=False)
        tables.append(network_table(an))
        for key, val in network_arrays(an).items():
            arrays.setdefault(key, []).append(val)
        name = an["condition"]["name"]
        for key in ("step", "loss", "acc_A", "acc_B", "norm_in_A", "norm_in_B"):
            history[f"{name}/{key}"] = np.asarray(an["history"][key])
    table = pd.concat(tables, ignore_index=True)
    table.insert(0, "network", table["condition"] + "/" + table["net"].astype(str))
    table.to_csv(out_dir / "networks.csv", index=False)
    n_full = int(table["lam_ctxA"].notna().sum()) if "lam_ctxA" in table else 0
    stacked = {k: np.concatenate(v) for k, v in arrays.items() if sum(len(x) for x in v) == len(table)}
    np.savez_compressed(out_dir / "arrays.npz", **stacked)
    np.savez_compressed(out_dir / "training_history.npz", **history)
    print(f"{len(table)} networks from {len(files)} conditions ({n_full} with dynamics) -> {out_dir / 'networks.csv'}")
    print(f"excluded (accuracy < {MIN_ACCURACY:.0%} on either task): {int((~table['included']).sum())}")

    snap_files = sorted((runs / "analysis" / "snapshots").glob("*.pt"))
    if snap_files:
        snaps = []
        for path in snap_files:
            an = torch.load(path, weights_only=False)
            t = network_table(an)
            t.insert(0, "epoch", int(path.stem.rsplit("_e", 1)[1]))
            snaps.append(t)
        # the final networks are the last point of every time course
        final = table[table["condition"].isin({s["condition"].iloc[0] for s in snaps})][snaps[0].columns.drop("epoch")].copy()
        final.insert(0, "epoch", final["train_steps"] // TrainConfig().steps_per_epoch - final["k"])
        pd.concat(snaps + [final], ignore_index=True).to_csv(out_dir / "snapshots.csv", index=False)
        print(f"{len(snap_files)} snapshot files -> {out_dir / 'snapshots.csv'}")

    if (runs / "full_k0_rho1.pt").exists() and (runs / "analysis" / "full_k0_rho1.pt").exists():
        np.savez_compressed(out_dir / "example_network.npz", **example_network(runs))
        print(f"example network -> {out_dir / 'example_network.npz'}")


if __name__ == "__main__":
    main()
