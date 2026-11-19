"""Pre-specified statistical analysis (Section 7 of the proposal) on the collected results.

    python scripts/run_stats.py          # results/networks.csv, arrays.npz -> results/stats.json, results/report.md

Networks whose final-choice accuracy is below 90% on either task are excluded (their number is
reported). Unless stated otherwise response-time measures use the threshold theta = 0.5; the other
thresholds, and the measures without the exclusion of premature responses, are reported as robustness
checks. Group means come with 95% bootstrap confidence intervals over networks.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

from stroop_rnn.config import K_LEVELS, MAIN_THRESHOLD, MIN_ACCURACY, RHO_LEVELS, RT_THRESHOLDS
from stroop_rnn.stats import bootstrap_ci, sign_flip_test, two_sample_permutation_test, wilcoxon_paired

ROOT = Path(__file__).resolve().parents[1]
TH = MAIN_THRESHOLD
THETA_INDEX = {th: i for i, th in enumerate(RT_THRESHOLDS)}
CSI_MS = (100, 200, 400, 800)
RCI_MS = (100, 200, 400, 600)
ARMS = ("repeat", "switch", "switch_to_target", "switch_to_neutral")
LONG_RCI_MS = (100, 600, 2000, 5000)
STEP_CSI = (100, 800)             # cue-stimulus intervals of the noise-free step responses
STEP_T = (50, 100, 200, 500)      # times after stimulus onset at which they are tabulated


# ------------------------------------------------------------------------------- helpers
def summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]                       # e.g. a time constant is infinite if the distance does not shrink
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "sd": None, "n": 0}
    mean, lo, hi = bootstrap_ci(v)
    return {"mean": float(mean), "lo": float(lo), "hi": float(hi), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "n": int(len(v))}


def msummary(values) -> dict:
    """Median with a bootstrap confidence interval of the median.

    Used for the quantities of the linearisation (effective gains, selection-vector fractions): they are
    ratios or products of eigenvectors, and single networks without a clean line attractor give extreme
    values that would dominate a mean.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "sd": None, "n": 0, "stat": "median"}
    med, lo, hi = bootstrap_ci(v, stat=np.median, n_boot=4000)
    return {"stat": "median", "median": float(med), "lo": float(lo), "hi": float(hi), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def fmt(s: dict, digits: int = 2) -> str:
    if not s or s.get("mean") is None:
        return "n/a"
    centre = s["median"] if s.get("stat") == "median" else s["mean"]
    return f"{centre:.{digits}f} [{s['lo']:.{digits}f}, {s['hi']:.{digits}f}]"


def pfmt(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "< .001" if p < 0.001 else f"{p:.3f}"


def col(df: pd.DataFrame, name: str, theta: float = TH, raw: bool = False) -> pd.Series:
    """A measure of the behavioural battery; error rates and rates of premature responses are returned in percent."""
    values = df[f"{name}@{theta}{':raw' if raw else ''}"]
    return 100.0 * values if name.startswith(("err_", "premature_")) else values


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["k_s"] = df["k"] / 10.0                       # per 10 pre-training epochs
    df["lr"] = np.log2(df["rho"])                    # per doubling of the task ratio
    df["norm"] = df["norm_control"].astype(int)
    for theta in RT_THRESHOLDS:
        for raw in (False, True):
            sfx = f"@{theta}{':raw' if raw else ''}"
            df[f"facilitation_asym{sfx}"] = df[f"facilitation_B{sfx}"] - df[f"facilitation_A{sfx}"]
            df[f"stroop_A{sfx}"] = df[f"rt_incong_A{sfx}"] - df[f"rt_cong_A{sfx}"]
            df[f"stroop_B{sfx}"] = df[f"rt_incong_B{sfx}"] - df[f"rt_cong_B{sfx}"]
            df[f"stroop_asym{sfx}"] = df[f"stroop_B{sfx}"] - df[f"stroop_A{sfx}"]
            df[f"rt_asym{sfx}"] = df[f"rt_B{sfx}"] - df[f"rt_A{sfx}"]
            df[f"err_stroop_A{sfx}"] = df[f"acc_cong_A{sfx}"] - df[f"acc_incong_A{sfx}"]
            df[f"err_stroop_B{sfx}"] = df[f"acc_cong_B{sfx}"] - df[f"acc_incong_B{sfx}"]
            df[f"err_stroop_asym{sfx}"] = df[f"err_stroop_B{sfx}"] - df[f"err_stroop_A{sfx}"]
    if "sel_fraction_A" in df:
        for tag in ("", "_ref", "_op"):
            for conv in ("", "_unit", "_dv"):
                a, b = f"sel_fraction{conv}_A{tag}", f"sel_fraction{conv}_B{tag}"
                df[f"sel_fraction{conv}_diff{tag}"] = df[a] - df[b]
        df["intrusion_asym"] = df["intrusion_on_B"] - df["intrusion_on_A"]
        df["psy_intrusion_asym"] = df["psy_intrusion_on_B"] - df["psy_intrusion_on_A"]
        df["gain_rel_dv_diff"] = df["gain_rel_dv_A"] - df["gain_rel_dv_B"]
        df["norm_in_diff"] = df["norm_in_A"] - df["norm_in_B"]
        df["line_attractor"] = (df["slow_mode_ctxA"].astype(bool) & df["slow_mode_ctxB"].astype(bool)
                                & df["others_decay_ctxA"].astype(bool) & df["others_decay_ctxB"].astype(bool))
    if "nf_interference_A_csi100" in df:
        # noise-free responses: "intrusion" = response to the irrelevant feature / response to the relevant feature
        for ms in STEP_CSI:
            df[f"nf_interference_asym_csi{ms}"] = df[f"nf_interference_B_csi{ms}"] - df[f"nf_interference_A_csi{ms}"]
            df[f"nf_rt_asym_csi{ms}"] = df[f"nf_rt_neut_B_csi{ms}"] - df[f"nf_rt_neut_A_csi{ms}"]
            for t in STEP_T:
                for task in "AB":
                    df[f"nf_intrusion_{task}_csi{ms}_t{t}"] = df[f"nf_irr_{task}_csi{ms}_t{t}"] / df[f"nf_rel_{task}_csi{ms}_t{t}"]
                df[f"nf_intrusion_asym_csi{ms}_t{t}"] = df[f"nf_intrusion_B_csi{ms}_t{t}"] - df[f"nf_intrusion_A_csi{ms}_t{t}"]
        df["nf_slope_rel_diff"] = 1000.0 * (df["nf_slope_rel_A"] - df["nf_slope_rel_B"])     # per second
    return df


def grid(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """The 9-cell k x rho grid of one variant (Adam, free input norms), included networks only."""
    m = (df["variant"] == variant) & (df["optimizer"] == "adam") & (~df["norm_control"]) & df["included"]
    return df[m]


def cell(df: pd.DataFrame, variant: str, k: int, rho: int, norm: bool = False, optimizer: str = "adam") -> pd.DataFrame:
    m = ((df["variant"] == variant) & (df["k"] == k) & (df["rho"] == rho) & (df["norm_control"] == norm)
         & (df["optimizer"] == optimizer) & df["included"])
    return df[m]


def ols(data: pd.DataFrame, formula: str) -> dict:
    """Ordinary least squares with heteroscedasticity-robust (HC3) standard errors."""
    data = data.dropna(subset=[formula.split("~")[0].strip()])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols(formula, data=data).fit(cov_type="HC3")
        ci = fit.conf_int()
        return {"formula": formula, "n": int(fit.nobs), "r2": float(fit.rsquared),
                "terms": {name: {"coef": float(fit.params[name]), "se": float(fit.bse[name]), "lo": float(ci.loc[name, 0]),
                                 "hi": float(ci.loc[name, 1]), "p": float(fit.pvalues[name])} for name in fit.params.index}}
    except Exception as exc:                                                    # e.g. not enough conditions
        return {"formula": formula, "error": str(exc)}


def qreg(data: pd.DataFrame, formula: str) -> dict:
    """Median (least absolute deviations) regression: the robust counterpart of `ols` for heavy-tailed measures."""
    data = data.dropna(subset=[formula.split("~")[0].strip()])
    data = data[np.isfinite(data[formula.split("~")[0].strip()])]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.quantreg(formula, data=data).fit(q=0.5)
        ci = fit.conf_int()
        return {"formula": formula, "method": "median regression", "n": int(fit.nobs),
                "terms": {name: {"coef": float(fit.params[name]), "se": float(fit.bse[name]), "lo": float(ci.loc[name, 0]),
                                 "hi": float(ci.loc[name, 1]), "p": float(fit.pvalues[name])} for name in fit.params.index}}
    except Exception as exc:
        return {"formula": formula, "error": str(exc)}


def term(model: dict, name: str) -> str:
    if "terms" not in model or name not in model["terms"]:
        return "n/a"
    t = model["terms"][name]
    return f"{t['coef']:.2f} [{t['lo']:.2f}, {t['hi']:.2f}], p {pfmt(t['p'])}"


# ------------------------------------------------------------------------ sections
def exclusions(df: pd.DataFrame) -> dict:
    out = {}
    for name, g in df.groupby("condition", sort=True):
        out[name] = {"n": int(len(g)), "excluded": int((~g["included"]).sum()), "min_acc": float(g["min_acc"].min()),
                     "mean_acc_A": float(g["final_acc_A"].mean()), "mean_acc_B": float(g["final_acc_B"].mean())}
    return {"criterion": f"final-choice accuracy >= {MIN_ACCURACY} on both tasks", "total": int(len(df)),
            "total_excluded": int((~df["included"]).sum()), "conditions": out}


def h1_replication(df: pd.DataFrame, variant: str) -> dict:
    g = cell(df, variant, 0, 1)
    if len(g) == 0:
        return {}
    out = {"n": int(len(g)), "accuracy_A": summary(g["final_acc_A"]), "accuracy_B": summary(g["final_acc_B"])}
    # context-dependent behaviour: logistic weights of the relevant and the irrelevant feature
    rel = pd.concat([g["psy_bA_ctxA"], g["psy_bB_ctxB"]])
    irr = pd.concat([g["psy_bB_ctxA"], g["psy_bA_ctxB"]])
    out["psy_relevant_weight"] = summary(rel)
    out["psy_irrelevant_weight"] = summary(irr)
    out["psy_irrelevant_over_relevant"] = summary(pd.concat([g["psy_intrusion_on_A"], g["psy_intrusion_on_B"]]))
    out["psy_rel_vs_irr_wilcoxon"] = wilcoxon_paired((g["psy_bA_ctxA"] + g["psy_bB_ctxB"]) / 2,
                                                     (g["psy_bB_ctxA"].abs() + g["psy_bA_ctxB"].abs()) / 2)
    # approximate line attractor (pre-specified criteria at the slow point on the decision boundary)
    for cue in "AB":
        out[f"lambda_ctx{cue}"] = summary(g[f"lam_ctx{cue}"])
        out[f"lambda_second_ctx{cue}"] = summary(g[f"lam_second_ctx{cue}"])
        out[f"frac_slow_mode_ctx{cue}"] = float(g[f"slow_mode_ctx{cue}"].astype(bool).mean())
        out[f"frac_others_decay_ctx{cue}"] = float(g[f"others_decay_ctx{cue}"].astype(bool).mean())
        out[f"frac_mode_is_slowest_ctx{cue}"] = float(g[f"mode_is_slowest_ctx{cue}"].astype(bool).mean())
        out[f"valley_speed_max_ctx{cue}"] = summary(g[f"speed_max_ctx{cue}"])
        out[f"valley_pc1_ctx{cue}"] = summary(g[f"valley_pc1_ctx{cue}"])
        out[f"dc_share_ctx{cue}"] = summary(g[f"dc_share_ctx{cue}"])
        out[f"n_fixed_points_ctx{cue}"] = {str(int(k)): int(v) for k, v in g[f"n_fixed_ctx{cue}"].value_counts().sort_index().items()}
        out[f"n_stable_ctx{cue}"] = {str(int(k)): int(v) for k, v in g[f"n_stable_ctx{cue}"].value_counts().sort_index().items()}
        out[f"lambda_op_ctx{cue}"] = summary(g[f"lam_op_ctx{cue}"])
    out["lambda_abs_max"] = float(np.max(np.abs(g[["lam_ctxA", "lam_ctxB"]].to_numpy())))
    out["frac_line_attractor_both_contexts"] = float(g["line_attractor"].mean())
    out["n_line_attractor"] = int(g["line_attractor"].sum())
    # selection mechanism
    for conv in ("", "_unit", "_dv"):
        for tag in ("", "_ref", "_op"):
            both = pd.concat([g[f"sel_fraction{conv}_A{tag}"], g[f"sel_fraction{conv}_B{tag}"]])
            out[f"sel_fraction{conv}{tag}"] = msummary(both)
            out[f"frac_sel_fraction{conv}{tag}_above_half"] = float((both > 0.5).mean())
    out["cos_selection_vectors"] = summary(g["cos_l"])
    out["cos_line_attractors"] = summary(g["cos_r"])
    out["cos_input_directions"] = summary(pd.concat([g["cos_b_A"], g["cos_b_B"]]))
    out["gain_relevant"] = summary(pd.concat([g["gain_rel_A"], g["gain_rel_B"]]))
    out["gain_irrelevant"] = summary(pd.concat([g["gain_irr_A"], g["gain_irr_B"]]))
    return out


def line_attractor_all(df: pd.DataFrame) -> dict:
    out = {}
    for name, g in df[df["included"] & (df["variant"] != "rank1")].groupby("condition", sort=True):
        out[name] = {"n": int(len(g)), "frac_line_attractor": float(g["line_attractor"].mean()),
                     "median_abs_lambda": float(np.median(np.abs(g[["lam_ctxA", "lam_ctxB"]].to_numpy()))),
                     "max_abs_lambda": float(np.max(np.abs(g[["lam_ctxA", "lam_ctxB"]].to_numpy()))),
                     "sel_fraction_A": msummary(g["sel_fraction_A"]), "sel_fraction_B": msummary(g["sel_fraction_B"])}
    return out


def v_rank_one(df: pd.DataFrame) -> dict:
    r1 = df[(df["variant"] == "rank1") & df["included"]]
    full = cell(df, "full", 0, 1)
    if len(r1) == 0:
        return {}
    out = {"n": int(len(r1)), "n_trained": int((df["variant"] == "rank1").sum()),
           "accuracy": summary(r1["min_acc"]), "cos_selection_vectors": summary(r1["cos_l"]),
           "min_cos_selection_vectors": float(r1["cos_l"].min()),
           # the axis of the selection vector (its sign is a convention tied to the sign of the readout)
           "min_abs_cos_selection_vectors": float(r1["cos_l"].abs().min()),
           "n_sign_flipped": int((r1["cos_l"] < 0).sum())}
    same = r1[r1["cos_l"] > 0]
    out["sel_fraction_unit_max_abs_same_sign"] = float(pd.concat([same["sel_fraction_unit_A"], same["sel_fraction_unit_B"]]).abs().max())
    out["sel_fraction_mante_max_abs_same_sign"] = float(pd.concat([same["sel_fraction_A"], same["sel_fraction_B"]]).abs().max())
    for conv, label in (("", "mante"), ("_unit", "unit"), ("_dv", "dv")):
        both = pd.concat([r1[f"sel_fraction{conv}_A"], r1[f"sel_fraction{conv}_B"]])
        out[f"sel_fraction_{label}"] = msummary(both)
        out[f"sel_fraction_{label}_max_abs"] = float(both.abs().max())
        if len(full):
            both_full = pd.concat([full[f"sel_fraction{conv}_A"], full[f"sel_fraction{conv}_B"]])
            out[f"sel_fraction_{label}_full_networks"] = msummary(both_full)
            out[f"sel_fraction_{label}_rank1_vs_full"] = two_sample_permutation_test(
                (r1[f"sel_fraction{conv}_A"] + r1[f"sel_fraction{conv}_B"]) / 2,
                (full[f"sel_fraction{conv}_A"] + full[f"sel_fraction{conv}_B"]) / 2)
    out["lambda"] = summary(pd.concat([r1["lam_ctxA"], r1["lam_ctxB"]]))
    out["interference"] = summary((col(r1, "interference_A") + col(r1, "interference_B")) / 2)
    return out


H2_MEASURES = ("interference_A", "interference_B", "interference_asym", "facilitation_A", "facilitation_B",
               "facilitation_asym", "stroop_A", "stroop_B", "stroop_asym", "err_interference_A", "err_interference_B",
               "err_interference_asym", "err_stroop_asym", "rt_A", "rt_B", "rt_asym")


def cell_table(df: pd.DataFrame, variant: str, measures, theta: float = TH) -> dict:
    out = {}
    for k in K_LEVELS:
        for rho in RHO_LEVELS:
            g = cell(df, variant, k, rho)
            if len(g):
                out[f"k{k}_rho{rho}"] = {m: summary(col(g, m, theta)) for m in measures}
    return out


def h2_interference(df: pd.DataFrame, variant: str) -> dict:
    g = grid(df, variant)
    if len(g) == 0:
        return {}
    out = {"n": int(len(g)), "cells": cell_table(df, variant, H2_MEASURES)}
    data = g[["k_s", "lr", "norm", "edim"]].copy()
    out["regression"] = {}
    for m in ("interference_asym", "interference_A", "interference_B", "facilitation_asym", "stroop_asym",
              "err_interference_asym", "err_stroop_asym", "rt_asym"):
        data["y"] = col(g, m).to_numpy()
        out["regression"][m] = ols(data, "y ~ k_s * lr")
    out["robustness"] = {}
    for theta in RT_THRESHOLDS:
        for raw in (False, True):
            data["y"] = col(g, "interference_asym", theta, raw).to_numpy()
            out["robustness"][f"theta{theta}{'_raw' if raw else ''}"] = ols(data, "y ~ k_s * lr")
    out["sign_flip_asym_by_cell"] = {
        f"k{k}_rho{rho}": sign_flip_test(col(cell(df, variant, k, rho), "interference_asym"))
        for k in K_LEVELS for rho in RHO_LEVELS if len(cell(df, variant, k, rho))}
    # input-norm control: corners of the grid, with and without the control
    corners = df[(df["variant"] == variant) & (df["optimizer"] == "adam") & df["included"]
                 & df["k"].isin([K_LEVELS[0], K_LEVELS[-1]]) & df["rho"].isin([RHO_LEVELS[0], RHO_LEVELS[-1]])]
    cdata = corners[["k_s", "lr", "norm"]].copy()
    out["norm_control"] = {"n_control": int(corners["norm"].sum()), "n_free": int((1 - corners["norm"]).sum())}
    for m in ("interference_asym", "stroop_asym", "err_interference_asym", "rt_asym"):
        cdata["y"] = col(corners, m).to_numpy()
        out["norm_control"][m] = {
            "with_interactions": ols(cdata, "y ~ (k_s + lr) * norm"),
            "control_only": ols(cdata[cdata["norm"] == 1], "y ~ k_s + lr"),
            "free_only": ols(cdata[cdata["norm"] == 0], "y ~ k_s + lr"),
        }
    out["norm_control"]["cells"] = {}
    for k in (K_LEVELS[0], K_LEVELS[-1]):
        for rho in (RHO_LEVELS[0], RHO_LEVELS[-1]):
            for norm in (False, True):
                c = cell(df, variant, k, rho, norm)
                if len(c):
                    out["norm_control"]["cells"][f"k{k}_rho{rho}{'_norm' if norm else ''}"] = {
                        "interference_asym": summary(col(c, "interference_asym")), "rt_asym": summary(col(c, "rt_asym")),
                        "norm_in_A": summary(c["norm_in_A"]), "norm_in_B": summary(c["norm_in_B"])}
    # does the linear analysis predict the behavioural asymmetry?
    if "intrusion_asym" in g:
        out["links"] = {}
        for x, y in (("intrusion_asym", "stroop_asym"), ("psy_intrusion_asym", "stroop_asym"),
                     ("intrusion_asym", "psy_intrusion_asym"), ("gain_rel_dv_diff", "rt_asym")):
            yv = col(g, y) if f"{y}@{TH}" in g else g[y]
            ok = ~(g[x].isna() | yv.isna())
            rho_s, p = sps.spearmanr(g[x][ok], yv[ok]) if ok.sum() > 3 else (np.nan, np.nan)
            out["links"][f"{x}~{y}"] = {"spearman": float(rho_s), "p": float(p), "n": int(ok.sum())}
    return out


def mechanism(df: pd.DataFrame, arrays: dict, variant: str) -> dict:
    """Where the interference comes from: noise-free responses to each feature, and interference against the CSI."""
    g = grid(df, variant)
    if len(g) == 0 or "nf_interference_A_csi100" not in g:
        return {}
    keys = [f"nf_{m}_{task}_csi{ms}" for m in ("interference", "facilitation", "rt_neut") for task in "AB" for ms in STEP_CSI]
    keys += [f"nf_intrusion_{task}_csi{ms}_t{t}" for task in "AB" for ms in STEP_CSI for t in STEP_T]
    keys += ["nf_slope_rel_A", "nf_slope_rel_B", "nf_slope_irr_A", "nf_slope_irr_B"]
    out = {"n": int(len(g)), "cells": {}}
    has_stroop = "stroop_rt" in arrays
    for k in K_LEVELS:
        for rho in RHO_LEVELS:
            c = cell(df, variant, k, rho)
            if len(c) == 0:
                continue
            entry = {key: summary(c[key]) for key in keys}
            for ms in STEP_CSI:
                entry[f"wilcoxon_nf_interference_csi{ms}_B_vs_A"] = wilcoxon_paired(c[f"nf_interference_B_csi{ms}"], c[f"nf_interference_A_csi{ms}"])
                entry[f"wilcoxon_nf_intrusion_csi{ms}_t50_B_vs_A"] = wilcoxon_paired(c[f"nf_intrusion_B_csi{ms}_t50"], c[f"nf_intrusion_A_csi{ms}_t50"])
            if has_stroop:
                rt = arrays["stroop_rt"][c.index.to_numpy()][:, THETA_INDEX[TH]]          # [n, task, congruency, CSI]
                interference = rt[:, :, 2] - rt[:, :, 1]                                   # [n, task, CSI]
                entry["interference_by_csi"] = {task: [summary(interference[:, ti, ci]) for ci in range(len(CSI_MS))]
                                                for ti, task in enumerate("AB")}
                entry["sign_flip_interference_csi100_vs_800"] = sign_flip_test(np.nanmean(interference[:, :, 0] - interference[:, :, 3], axis=1))
                asym = interference[:, 1] - interference[:, 0]
                entry["interference_asym_by_csi"] = [summary(asym[:, ci]) for ci in range(len(CSI_MS))]
            out["cells"][f"k{k}_rho{rho}"] = entry
    data = g[["k_s", "lr"]].copy()
    out["regression"] = {}
    names = [f"nf_interference_asym_csi{ms}" for ms in STEP_CSI] + [f"nf_rt_asym_csi{ms}" for ms in STEP_CSI]
    names += [f"nf_intrusion_asym_csi{ms}_t{t}" for ms in STEP_CSI for t in (50, 500)]
    names += [f"nf_intrusion_{task}_csi100_t50" for task in "AB"] + ["nf_slope_rel_diff"]
    for name in names:
        data["y"] = g[name].to_numpy()
        out["regression"][name] = ols(data, "y ~ k_s * lr")
    if has_stroop:
        rt = arrays["stroop_rt"][g.index.to_numpy()][:, THETA_INDEX[TH]]
        interference = rt[:, :, 2] - rt[:, :, 1]
        for ci, ms in enumerate(CSI_MS):
            data["y"] = interference[:, 1, ci] - interference[:, 0, ci]
            out["regression"][f"interference_asym_csi{ms}"] = ols(data, "y ~ k_s * lr")
        out["sign_flip_interference_csi100_vs_800"] = sign_flip_test(np.nanmean(interference[:, :, 0] - interference[:, :, 3], axis=1))
        out["interference_by_csi_all"] = [summary(np.nanmean(interference[:, :, ci], axis=1)) for ci in range(len(CSI_MS))]
    out["links"] = {}
    for x, y in (("nf_interference_asym_csi100", f"interference_asym@{TH}"), ("nf_interference_asym_csi800", f"interference_asym@{TH}"),
                 ("nf_intrusion_asym_csi100_t50", f"interference_asym@{TH}"), ("nf_rt_asym_csi800", f"rt_asym@{TH}"),
                 ("nf_slope_rel_diff", f"rt_asym@{TH}"), ("nf_slope_rel_A", "nf_slope_meas_rel_A"), ("nf_slope_irr_A", "nf_slope_meas_irr_A")):
        ok = ~(g[x].isna() | g[y].isna())
        r, p = sps.spearmanr(g[x][ok], g[y][ok]) if ok.sum() > 3 else (np.nan, np.nan)
        out["links"][f"{x} ~ {y}"] = {"spearman": float(r), "p": float(p), "n": int(ok.sum())}
    return out


def h3_selection(df: pd.DataFrame, variant: str) -> dict:
    g = grid(df, variant)
    if len(g) == 0:
        return {}
    out = {"n": int(len(g)), "cells": {}}
    keys = ("sel_fraction_A", "sel_fraction_B", "sel_fraction_unit_A", "sel_fraction_unit_B", "sel_fraction_dv_A",
            "sel_fraction_dv_B", "sel_mod_A", "sel_mod_B", "inp_mod_A", "inp_mod_B", "gain_rel_A", "gain_rel_B",
            "gain_irr_A", "gain_irr_B", "gain_rel_dv_A", "gain_rel_dv_B", "gain_irr_dv_A", "gain_irr_dv_B",
            "norm_in_A", "norm_in_B", "edim", "intrusion_on_A", "intrusion_on_B", "psy_intrusion_on_A", "psy_intrusion_on_B")
    for k in K_LEVELS:
        for rho in RHO_LEVELS:
            c = cell(df, variant, k, rho)
            if len(c) == 0:
                continue
            entry = {key: (summary if key in ("norm_in_A", "norm_in_B", "edim") else msummary)(c[key]) for key in keys}
            for conv in ("", "_unit", "_dv"):
                for tag in ("", "_ref", "_op"):
                    entry[f"wilcoxon_sel_fraction{conv}{tag}_A_vs_B"] = wilcoxon_paired(
                        c[f"sel_fraction{conv}_A{tag}"], c[f"sel_fraction{conv}_B{tag}"])
            entry["wilcoxon_inp_mod_A_vs_B"] = wilcoxon_paired(c["inp_mod_A"], c["inp_mod_B"])
            entry["wilcoxon_gain_rel_dv_A_vs_B"] = wilcoxon_paired(c["gain_rel_dv_A"], c["gain_rel_dv_B"])
            out["cells"][f"k{k}_rho{rho}"] = entry
    data = g[["k_s", "lr", "edim"]].copy()
    data["edim_c"] = data["edim"] - data["edim"].mean()
    out["regression"], out["regression_no_covariate"] = {}, {}
    names = [f"sel_fraction{conv}_diff{tag}" for conv in ("", "_unit", "_dv") for tag in ("", "_ref", "_op")]
    names += ["sel_fraction_A", "sel_fraction_B", "inp_mod_A", "inp_mod_B", "sel_mod_A", "sel_mod_B",
              "gain_rel_dv_diff", "norm_in_diff", "intrusion_asym", "psy_intrusion_asym"]
    for name in names:
        data["y"] = g[name].to_numpy()
        fit = ols if name == "norm_in_diff" else qreg          # median regression for the quantities of the linearisation
        out["regression"][name] = fit(data, "y ~ k_s + lr + edim_c")
        out["regression_no_covariate"][name] = fit(data, "y ~ k_s * lr")
    data["y"] = g["edim"].to_numpy()
    out["regression_no_covariate"]["edim"] = ols(data, "y ~ k_s * lr")
    out["wilcoxon_all_grid_sel_fraction_A_vs_B"] = wilcoxon_paired(g["sel_fraction_A"], g["sel_fraction_B"])
    over = g[(g["k"] > 0) | (g["rho"] > 1)]
    out["wilcoxon_overlearned_sel_fraction_A_vs_B"] = wilcoxon_paired(over["sel_fraction_A"], over["sel_fraction_B"])
    out["sel_fraction_diff_overlearned"] = msummary(over["sel_fraction_diff"])
    out["median_sel_fraction_diff_overlearned"] = float(np.nanmedian(over["sel_fraction_diff"])) if len(over) else None
    # perturbations along the selection vector vs random directions of the same norm
    out["perturbation"] = {
        "sensitivity_selection": summary(g["pert_sens_sel"]), "sensitivity_line": summary(g["pert_sens_line"]),
        "sensitivity_random": summary(g["pert_sens_rand"]), "predicted_selection": summary(g["pert_pred_sel"]),
        "ratio_selection_to_random": summary(g["pert_sens_sel"] / g["pert_sens_rand"]),
        "wilcoxon_selection_vs_random": wilcoxon_paired(g["pert_sens_sel"], g["pert_sens_rand"]),
        "wilcoxon_selection_vs_line": wilcoxon_paired(g["pert_sens_sel"], g["pert_sens_line"]),
        "measured_over_predicted": summary(g["pert_sens_sel"] / g["pert_pred_sel"]),
        "spearman_predicted_measured": float(sps.spearmanr(g["pert_pred_sel"], g["pert_sens_sel"])[0]),
        "choice_bias": {name: {a: summary(g[f"pert_bias_{name}_a{a}"]) for a in ("1", "2", "4", "8")}
                        for name in ("sel", "line", "rand")},
    }
    return out


def mixed_model(df: pd.DataFrame, arrays: dict, index: np.ndarray, theta: float = TH) -> dict:
    """RT ~ transition x task x (k, log rho) + transition x (CSI, RCI), random intercept per network."""
    rt = arrays["cell_rt"][index][:, THETA_INDEX[theta]]            # [n, transition, task, CSI, RCI]
    n = len(index)
    sw, task, ci, ri = np.meshgrid(np.arange(2), np.arange(2), np.arange(4), np.arange(4), indexing="ij")
    rows = pd.DataFrame({
        "rt": rt.reshape(n, -1).ravel(),
        "network": np.repeat(df["network"].to_numpy(), sw.size),
        "k_s": np.repeat(df["k_s"].to_numpy(), sw.size), "lr": np.repeat(df["lr"].to_numpy(), sw.size),
        "sw": np.tile(sw.ravel() - 0.5, n),                         # +0.5 = switch
        "task": np.tile(0.5 - task.ravel(), n),                     # +0.5 = task A, so sw:task = cost(A) - cost(B)
        "csi": np.tile(np.log2(np.asarray(CSI_MS) / 100.0)[ci.ravel()], n),
        "rci": np.tile(np.log2(np.asarray(RCI_MS) / 100.0)[ri.ravel()], n),
    }).dropna(subset=["rt"])
    rows["csi"] -= np.log2(np.asarray(CSI_MS) / 100.0).mean()
    rows["rci"] -= np.log2(np.asarray(RCI_MS) / 100.0).mean()
    formula = "rt ~ sw * task * (k_s + lr) + sw * (csi + rci)"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(formula, rows, groups=rows["network"]).fit(reml=True)
        ci = fit.conf_int()
        terms = {name: {"coef": float(fit.params[name]), "se": float(fit.bse[name]), "lo": float(ci.loc[name, 0]),
                        "hi": float(ci.loc[name, 1]), "p": float(fit.pvalues[name])} for name in fit.params.index}
        return {"formula": formula + " + (1 | network)", "n_rows": int(len(rows)), "n_networks": int(rows["network"].nunique()),
                "converged": bool(fit.converged), "terms": terms,
                "coding": "sw: +0.5 switch / -0.5 repeat; task: +0.5 A / -0.5 B; csi, rci: centred log2(ms / 100); "
                          "k_s: pre-training epochs / 10; lr: log2 rho"}
    except Exception as exc:
        return {"formula": formula, "error": str(exc)}


def h4_switch(df: pd.DataFrame, arrays: dict, variant: str) -> dict:
    g = grid(df, variant)
    if len(g) == 0:
        return {}
    measures = ("switch_cost_A", "switch_cost_B", "switch_asym", "err_switch_cost_A", "err_switch_cost_B", "err_switch_asym",
                "premature_repeat_A", "premature_repeat_B", "premature_switch_A", "premature_switch_B")
    out = {"n": int(len(g)), "cells": cell_table(df, variant, measures)}
    for key, c in out["cells"].items():
        k, rho = (int(v) for v in key.replace("k", "").split("_rho"))
        cc = cell(df, variant, k, rho)
        c["sign_flip_cost_A"] = sign_flip_test(col(cc, "switch_cost_A"))
        c["sign_flip_cost_B"] = sign_flip_test(col(cc, "switch_cost_B"))
        c["sign_flip_asym"] = sign_flip_test(col(cc, "switch_asym"))
        rt = arrays["cell_rt"][cc.index.to_numpy()][:, THETA_INDEX[TH]]            # [n, transition, task, CSI, RCI]
        cost = rt[:, 1] - rt[:, 0]                                                  # [n, task, CSI, RCI]
        c["cost_by_csi"] = np.nanmean(np.nanmean(cost, axis=3), axis=0).tolist()    # [task, CSI]
        c["cost_by_rci"] = np.nanmean(np.nanmean(cost, axis=2), axis=0).tolist()    # [task, RCI]
        c["cost_shortest"] = {"A": summary(cost[:, 0, 0, 0]), "B": summary(cost[:, 1, 0, 0])}
        c["cost_csi100"] = {"A": summary(np.nanmean(cost[:, 0, 0], axis=1)), "B": summary(np.nanmean(cost[:, 1, 0], axis=1))}
        c["cost_csi800"] = {"A": summary(np.nanmean(cost[:, 0, 3], axis=1)), "B": summary(np.nanmean(cost[:, 1, 3], axis=1))}
        c["sign_flip_csi_effect"] = sign_flip_test(np.nanmean(cost[:, :, 0] - cost[:, :, 3], axis=(1, 2)))
    data = g[["k_s", "lr"]].copy()
    out["regression"] = {}
    for m in ("switch_asym", "switch_cost_A", "switch_cost_B", "err_switch_asym"):
        data["y"] = col(g, m).to_numpy()
        out["regression"][m] = ols(data, "y ~ k_s * lr")
    out["robustness"] = {}
    for theta in RT_THRESHOLDS:
        for raw in (False, True):
            data["y"] = col(g, "switch_asym", theta, raw).to_numpy()
            out["robustness"][f"theta{theta}{'_raw' if raw else ''}"] = ols(data, "y ~ k_s * lr")
    out["mixed_model"] = mixed_model(g, arrays, g.index.to_numpy())
    out["mixed_model_other_thresholds"] = {str(th): mixed_model(g, arrays, g.index.to_numpy(), th)
                                           for th in RT_THRESHOLDS if th != TH}
    # time for the state to move between the two contexts' structures, against the slowest eigenvalues
    rel = {}
    for start in ("switch", "rest"):
        for key in ("t_half", "t_e", "tau_tail", "dist0"):
            rel[f"{key}_{start}"] = summary(pd.concat([g[f"relax_{key}_{start}_A"], g[f"relax_{key}_{start}_B"]]))
    tau_eig = pd.concat([g["relax_tau_eig2_A"], g["relax_tau_eig2_B"]]).to_numpy()
    rel["tau_eig2"] = summary(tau_eig[np.isfinite(tau_eig)])
    for start in ("switch", "rest"):
        tail = pd.concat([g[f"relax_tau_tail_{start}_A"], g[f"relax_tau_tail_{start}_B"]]).to_numpy()
        ok = np.isfinite(tail) & np.isfinite(tau_eig)
        r, p = sps.spearmanr(tau_eig[ok], tail[ok]) if ok.sum() > 3 else (np.nan, np.nan)
        rel[f"spearman_tau_tail_{start}_vs_tau_eig2"] = {"spearman": float(r), "p": float(p), "n": int(ok.sum())}
        rel[f"ratio_tau_tail_{start}_to_tau_eig2"] = summary(tail[ok] / tau_eig[ok])
    idx = g.index.to_numpy()
    rt = arrays["cell_rt"][idx][:, THETA_INDEX[TH]]
    cost_short = np.nanmean((rt[:, 1] - rt[:, 0])[:, :, 0, :2], axis=(1, 2))        # CSI 100 ms, RCI 100-200 ms
    t_e = (g["relax_t_e_switch_A"] + g["relax_t_e_switch_B"]).to_numpy() / 2
    ok = np.isfinite(cost_short) & np.isfinite(t_e)
    r, p = sps.spearmanr(t_e[ok], cost_short[ok]) if ok.sum() > 3 else (np.nan, np.nan)
    rel["spearman_t_e_switch_vs_cost_short_csi"] = {"spearman": float(r), "p": float(p), "n": int(ok.sum())}
    rel["dist_at_csi_switch"] = np.nanmean(arrays["relax_dist_csi"][idx][:, :, 0], axis=(0, 1)).tolist()
    rel["dist_at_csi_rest"] = np.nanmean(arrays["relax_dist_csi"][idx][:, :, 1], axis=(0, 1)).tolist()
    out["relaxation"] = rel
    return out


def e5_condition(arrays: dict, idx: np.ndarray, theta: float = TH) -> dict:
    """State-reset experiment for one group of networks. Arms: repeat, switch, switch->target, switch->neutral."""
    ti = THETA_INDEX[theta]
    rt = arrays["e5_rt"][idx][:, ti]                      # [n, arm, task, CSI]
    R, S, T, N = (rt[:, i] for i in range(4))             # each [n, task, CSI]
    out = {"n": int(len(idx)), "rt_mean": np.nanmean(rt, axis=0).tolist(),
           "premature_mean": np.nanmean(arrays["e5_premature"][idx][:, ti], axis=0).tolist(),
           "acc_mean": np.nanmean(arrays["e5_acc"][idx][:, ti], axis=0).tolist()}

    def describe(x: np.ndarray) -> dict:                  # x: [n, task, CSI]
        asym = x[:, 0] - x[:, 1]
        return {
            "A": summary(np.nanmean(x[:, 0], axis=1)), "B": summary(np.nanmean(x[:, 1], axis=1)),
            "A_csi100": summary(x[:, 0, 0]), "B_csi100": summary(x[:, 1, 0]),
            "A_csi800": summary(x[:, 0, 3]), "B_csi800": summary(x[:, 1, 3]),
            "by_csi": np.nanmean(x, axis=0).tolist(),
            "test_mean": sign_flip_test(np.nanmean(x, axis=(1, 2))),
            "test_csi100": sign_flip_test(np.nanmean(x[:, :, 0], axis=1)),
            "test_csi800": sign_flip_test(np.nanmean(x[:, :, 3], axis=1)),
            "test_csi_effect": sign_flip_test(np.nanmean(x[:, :, 0] - x[:, :, 3], axis=1)),
            "asym": summary(np.nanmean(asym, axis=1)), "test_asym": sign_flip_test(np.nanmean(asym, axis=1)),
            "asym_csi100": summary(asym[:, 0]), "test_asym_csi100": sign_flip_test(asym[:, 0]),
        }

    out["cost_none"] = describe(S - R)
    out["cost_target"] = describe(T - R)
    out["cost_neutral"] = describe(N - R)
    out["target_vs_none"] = describe(T - S)
    out["neutral_vs_none"] = describe(N - S)
    # additive split of the switch cost: (S - N) + (N - T) + (T - R)
    out["inertia"] = describe(S - N)
    out["reconfiguration"] = describe(N - T)
    out["repetition_benefit"] = describe(T - R)
    # the same costs without the exclusion of premature responses
    raw = arrays["e5_rt_raw"][idx][:, ti]
    out["raw"] = {name: describe(raw[:, a] - raw[:, b]) for name, a, b in
                  (("cost_none", 1, 0), ("cost_target", 2, 0), ("cost_neutral", 3, 0), ("inertia", 1, 3), ("reconfiguration", 3, 2))}
    # response times at the shortest CSI in every arm
    prem = arrays["e5_premature"][idx][:, ti]
    acc = arrays["e5_acc"][idx][:, ti]
    out["csi100"] = {arm: {"rt_A": summary(rt[:, i, 0, 0]), "rt_B": summary(rt[:, i, 1, 0]),
                           "premature_A": float(np.nanmean(prem[:, i, 0, 0])), "premature_B": float(np.nanmean(prem[:, i, 1, 0])),
                           "acc_A": float(np.nanmean(acc[:, i, 0, 0])), "acc_B": float(np.nanmean(acc[:, i, 1, 0]))}
                     for i, arm in enumerate(ARMS)}
    # repeat trials after long response-cue intervals
    lr = arrays["long_rt"][idx][:, ti]                    # [n, task, RCI, CSI]
    lprem = arrays["long_premature"][idx][:, ti]
    out["long_rsi"] = {"rt_mean": np.nanmean(lr, axis=0).tolist(),
                       "premature_mean": np.nanmean(lprem, axis=0).tolist(),
                       "csi100": {f"rci{ms}": {"rt_A": summary(lr[:, 0, ri, 0]), "rt_B": summary(lr[:, 1, ri, 0]),
                                               "premature_A": float(np.nanmean(lprem[:, 0, ri, 0])),
                                               "premature_B": float(np.nanmean(lprem[:, 1, ri, 0]))}
                                  for ri, ms in enumerate(LONG_RCI_MS)}}
    for label, ri in (("5000_vs_100", 3), ("2000_vs_100", 2), ("600_vs_100", 1)):
        cost = lr[:, :, ri] - lr[:, :, 0]                 # [n, task, CSI]
        out["long_rsi"][label] = {
            "A_csi100": summary(cost[:, 0, 0]), "B_csi100": summary(cost[:, 1, 0]),
            "A_csi400": summary(cost[:, 0, 1]), "B_csi400": summary(cost[:, 1, 1]),
            "test_cost_csi100": sign_flip_test(np.nanmean(cost[:, :, 0], axis=1)),
            "asym_csi100": summary(cost[:, 0, 0] - cost[:, 1, 0]), "test_asym_csi100": sign_flip_test(cost[:, 0, 0] - cost[:, 1, 0]),
            "asym_csi400": summary(cost[:, 0, 1] - cost[:, 1, 1]), "test_asym_csi400": sign_flip_test(cost[:, 0, 1] - cost[:, 1, 1]),
        }
    beyond = lr[:, :, 3] - lr[:, :, 1]                    # 5000 ms vs 600 ms: both beyond the reach of the previous trial?
    out["long_rsi"]["5000_vs_600"] = {"A_csi100": summary(beyond[:, 0, 0]), "B_csi100": summary(beyond[:, 1, 0]),
                                      "asym_csi100": summary(beyond[:, 0, 0] - beyond[:, 1, 0]),
                                      "test_asym_csi100": sign_flip_test(beyond[:, 0, 0] - beyond[:, 1, 0])}
    return out


def h5_interventions(df: pd.DataFrame, arrays: dict, variant: str) -> dict:
    out = {"conditions": {}}
    for k in K_LEVELS:
        for rho in RHO_LEVELS:
            c = cell(df, variant, k, rho)
            if len(c):
                out["conditions"][f"k{k}_rho{rho}"] = e5_condition(arrays, c.index.to_numpy())
    g = grid(df, variant)
    if len(g) == 0:
        return out
    # how the asymmetries depend on the learning history
    rt = arrays["e5_rt"][g.index.to_numpy()][:, THETA_INDEX[TH]]
    R, S, T, N = (rt[:, i] for i in range(4))
    lr = arrays["long_rt"][g.index.to_numpy()][:, THETA_INDEX[TH]]
    data = g[["k_s", "lr"]].copy()
    out["regression"] = {}
    for name, x in (("cost_none", S - R), ("cost_target", T - R), ("cost_neutral", N - R), ("inertia", S - N),
                    ("reconfiguration", N - T), ("repetition_benefit", T - R)):
        data["y"] = np.nanmean(x[:, 0] - x[:, 1], axis=1)
        out["regression"][f"asym_{name}"] = ols(data, "y ~ k_s * lr")
        data["y"] = (x[:, 0, 0] - x[:, 1, 0])
        out["regression"][f"asym_{name}_csi100"] = ols(data, "y ~ k_s * lr")
    cost = lr[:, :, 3] - lr[:, :, 0]
    data["y"] = cost[:, 0, 0] - cost[:, 1, 0]
    out["regression"]["asym_long_rsi_csi100"] = ols(data, "y ~ k_s * lr")
    # the two ends of the learning-history manipulation, compared directly
    lo, hi = cell(df, variant, K_LEVELS[0], RHO_LEVELS[0]), cell(df, variant, K_LEVELS[-1], RHO_LEVELS[-1])
    if len(lo) and len(hi):
        out["baseline_vs_overlearned"] = {}
        for name, a, b in (("cost_none", 1, 0), ("cost_target", 2, 0), ("cost_neutral", 3, 0), ("inertia", 1, 3),
                           ("reconfiguration", 3, 2)):
            def asym(c):
                r = arrays["e5_rt"][c.index.to_numpy()][:, THETA_INDEX[TH]]
                x = r[:, a] - r[:, b]
                return np.nanmean(x[:, 0] - x[:, 1], axis=1)
            out["baseline_vs_overlearned"][f"asym_{name}"] = two_sample_permutation_test(asym(hi), asym(lo))
    return out


H6_MEASURES = ("interference_asym", "stroop_asym", "err_interference_asym", "rt_asym", "switch_asym", "switch_cost_A", "switch_cost_B")
H6_DYNAMICS = ("sel_fraction_A", "sel_fraction_B", "sel_fraction_diff", "lam_ctxA", "lam_ctxB", "line_attractor", "intrusion_asym")


def h6_robustness(df: pd.DataFrame) -> dict:
    out = {"optimizer": {}, "dale_vs_full": {}}
    k_hi, rho_hi = K_LEVELS[-1], RHO_LEVELS[-1]
    for variant in ("full", "dale"):
        entry = {}
        groups = {"adam_base": cell(df, variant, 0, 1), "adam_over": cell(df, variant, k_hi, rho_hi),
                  "sgd_base": cell(df, variant, 0, 1, optimizer="sgd"), "sgd_over": cell(df, variant, k_hi, rho_hi, optimizer="sgd")}
        if any(len(v) == 0 for v in groups.values()):
            continue
        entry["n"] = {name: int(len(v)) for name, v in groups.items()}
        entry["accuracy"] = {name: summary(v["min_acc"]) for name, v in groups.items()}
        for m in H6_MEASURES + H6_DYNAMICS:
            get = (lambda v, m=m: col(v, m)) if m in H6_MEASURES else (lambda v, m=m: v[m].astype(float))
            entry[m] = {
                **{name: (msummary if m.startswith(("sel_fraction", "intrusion")) else summary)(get(v)) for name, v in groups.items()},
                "overlearning_effect_adam": two_sample_permutation_test(get(groups["adam_over"]), get(groups["adam_base"])),
                "overlearning_effect_sgd": two_sample_permutation_test(get(groups["sgd_over"]), get(groups["sgd_base"])),
                "sgd_vs_adam_base": two_sample_permutation_test(get(groups["sgd_base"]), get(groups["adam_base"])),
                "sgd_vs_adam_over": two_sample_permutation_test(get(groups["sgd_over"]), get(groups["adam_over"])),
            }
        out["optimizer"][variant] = entry
    both = df[df["variant"].isin(["full", "dale"]) & (df["optimizer"] == "adam") & (~df["norm_control"]) & df["included"]].copy()
    if both["variant"].nunique() == 2:
        both["dale"] = (both["variant"] == "dale").astype(int)
        for m in H6_MEASURES:
            both["y"] = col(both, m)
            out["dale_vs_full"][m] = ols(both, "y ~ (k_s + lr) * dale")
        for m in ("sel_fraction_diff", "intrusion_asym"):
            both["y"] = both[m]
            out["dale_vs_full"][m] = qreg(both, "y ~ (k_s + lr) * dale")
    return out


def snapshots(results: Path) -> dict:
    path = results / "snapshots.csv"
    if not path.exists():
        return {}
    snap = add_derived(pd.read_csv(path))
    snap = snap[snap["variant"].isin(["full", "dale"])]
    out = {}
    for (name, epoch), g in snap.groupby(["condition", "epoch"], sort=True):
        ok = g[g["min_acc"] >= MIN_ACCURACY]
        out.setdefault(name, {})[str(int(epoch))] = {
            "n": int(len(g)), "n_above_criterion": int(len(ok)), "acc_A": summary(g["final_acc_A"]), "acc_B": summary(g["final_acc_B"]),
            "interference_asym": summary(col(ok, "interference_asym")), "stroop_asym": summary(col(ok, "stroop_asym")),
            "rt_asym": summary(col(ok, "rt_asym")), "switch_asym": summary(col(ok, "switch_asym")),
            "interference_A": summary(col(ok, "interference_A")), "interference_B": summary(col(ok, "interference_B")),
        }
    return out


# ---------------------------------------------------------------------------------- report
def report(st: dict) -> str:
    L: list[str] = ["# Statistical report", "",
                    "Generated by `scripts/run_stats.py`. Values are means over networks with 95% bootstrap confidence intervals "
                    f"in brackets; response times in ms at the threshold theta = {TH} unless noted; error rates (`err_*`) and rates of "
                    "premature responses in percent. Task A is the task that is overlearned (k epochs of pre-training, rho times as many "
                    "trials as task B during joint training). Quantities of the linearisation (effective gains, selection-vector "
                    "fractions, input modulation) are summarised by medians with bootstrap intervals and analysed with median "
                    "regressions, because single networks without a clean line attractor give extreme ratios.", ""]
    ex = st["exclusions"]
    L += ["## Networks and exclusions", "",
          f"{ex['total']} networks were trained; {ex['total_excluded']} were excluded ({ex['criterion']}).", "",
          "| condition | n | excluded | lowest accuracy | mean accuracy A | mean accuracy B |", "|---|---|---|---|---|---|"]
    for name, c in ex["conditions"].items():
        L.append(f"| {name} | {c['n']} | {c['excluded']} | {c['min_acc']:.3f} | {c['mean_acc_A']:.3f} | {c['mean_acc_B']:.3f} |")

    for variant in ("full", "dale"):
        h1 = st["H1"].get(variant)
        if not h1:
            continue
        L += ["", f"## H1 / E1: replication in the baseline networks ({variant}, k = 0, rho = 1; n = {h1['n']})", "",
              f"- accuracy: task A {fmt(h1['accuracy_A'], 3)}, task B {fmt(h1['accuracy_B'], 3)}",
              f"- logistic weight of the relevant feature {fmt(h1['psy_relevant_weight'])}, of the irrelevant feature "
              f"{fmt(h1['psy_irrelevant_weight'], 3)} (ratio {fmt(h1['psy_irrelevant_over_relevant'], 3)}); "
              f"Wilcoxon relevant vs |irrelevant|: p {pfmt(h1['psy_rel_vs_irr_wilcoxon']['p'])}",
              f"- decision-mode eigenvalue (lambda tau) at the slow point on the decision boundary: context A {fmt(h1['lambda_ctxA'], 3)}, "
              f"context B {fmt(h1['lambda_ctxB'], 3)}; largest |lambda tau| {h1['lambda_abs_max']:.3f}",
              f"- next eigenvalue (largest real part of the rest): context A {fmt(h1['lambda_second_ctxA'], 3)}, context B {fmt(h1['lambda_second_ctxB'], 3)}",
              f"- pre-specified criteria (|lambda tau| < 0.1 and all other modes decaying) met in both contexts by "
              f"{h1['n_line_attractor']}/{h1['n']} networks ({100 * h1['frac_line_attractor_both_contexts']:.0f}%)",
              f"- slow points lie on a line: variance on the first principal component {fmt(h1['valley_pc1_ctxA'], 3)} (A), "
              f"{fmt(h1['valley_pc1_ctxB'], 3)} (B); largest speed tau |F| on the central part {fmt(h1['valley_speed_max_ctxA'], 3)} (A), "
              f"{fmt(h1['valley_speed_max_ctxB'], 3)} (B)",
              f"- exact fixed points per network: context A {h1['n_fixed_points_ctxA']}, context B {h1['n_fixed_points_ctxB']} "
              f"(stable: {h1['n_stable_ctxA']}, {h1['n_stable_ctxB']})",
              f"- share of the static gain of the relevant input carried by the decision mode: {fmt(h1['dc_share_ctxA'])} (A), {fmt(h1['dc_share_ctxB'])} (B)",
              f"- effective gain g = l . b: relevant {fmt(h1['gain_relevant'])}, irrelevant {fmt(h1['gain_irrelevant'])}",
              f"- selection-vector fraction of the gain change (Eq. 5): {fmt(h1['sel_fraction'], 3)} "
              f"({100 * h1['frac_sel_fraction_above_half']:.0f}% of feature x network cases above 0.5); "
              f"unit-norm selection vectors {fmt(h1['sel_fraction_unit'], 3)}; decision-variable units {fmt(h1['sel_fraction_dv'], 3)}",
              f"- same at the single slow point d = 0: {fmt(h1['sel_fraction_ref'], 3)}; at the operating point: {fmt(h1['sel_fraction_op'], 3)}",
              f"- cosine between the two contexts' selection vectors {fmt(h1['cos_selection_vectors'], 3)}, line attractors "
              f"{fmt(h1['cos_line_attractors'], 3)}, input directions of a feature {fmt(h1['cos_input_directions'], 3)}"]

    v = st.get("V")
    r1e = (st.get("E0") or {}).get("rank_one", {}).get("trained_networks")
    r1e = r1e if isinstance(r1e, dict) and "n_readout_sign_differs_between_contexts" in r1e else None
    if v:
        L += ["", f"## V: rank-one networks (n = {v['n']} of {v['n_trained']} above the accuracy criterion)", "",
              f"- accuracy {fmt(v['accuracy'], 3)}; smallest |cosine| between the selection vectors of the two contexts "
              f"{v['min_abs_cos_selection_vectors']:.6f} (the selection vector keeps its axis in every network)",
              f"- networks in which the two selection vectors have opposite signs (cosine -1): {v['n_sign_flipped']} of {v['n']}. The sign "
              "follows the sign of the readout along the line attractor; these networks read their single latent variable with opposite "
              "signs in the two contexts"
              + (f" (readout sign differs in {r1e['n_readout_sign_differs_between_contexts']} networks; same networks: "
                 f"{r1e['sign_flip_explained_by_readout_sign']})" if r1e else ""),
              f"- networks with equal signs: largest |selection-vector fraction| {v['sel_fraction_unit_max_abs_same_sign']:.1e} (unit-norm "
              f"selection vectors), {v['sel_fraction_mante_max_abs_same_sign']:.4f} (Mante normalisation)"
              + (f"; all networks after orienting both selection vectors the same way: {r1e['sel_fraction_unit_aligned_max_abs']:.1e} and "
                 f"{r1e['sel_fraction_mante_aligned_max_abs']:.4f}" if r1e else ""),
              "- with the readout-based sign convention, all networks (medians):",
              f"- selection-vector fraction, unit-norm selection vectors (rotation only): {fmt(v['sel_fraction_unit'], 6)}, largest |value| {v['sel_fraction_unit_max_abs']:.2e}",
              f"- selection-vector fraction, Mante normalisation: {fmt(v['sel_fraction_mante'], 4)}, largest |value| {v['sel_fraction_mante_max_abs']:.4f}"
              + (f" (full networks: {fmt(v['sel_fraction_mante_full_networks'], 3)}; permutation test p {pfmt(v['sel_fraction_mante_rank1_vs_full']['p'])})"
                 if "sel_fraction_mante_full_networks" in v else ""),
              f"- selection-vector fraction, decision-variable units: {fmt(v['sel_fraction_dv'], 4)}, largest |value| {v['sel_fraction_dv_max_abs']:.4f}",
              f"- decision-mode eigenvalue {fmt(v['lambda'], 3)}"]

    def cells_table(cells: dict, columns: list[tuple[str, str]], digits: int = 2) -> list[str]:
        rows = ["| condition | " + " | ".join(label for _, label in columns) + " |", "|---|" + "---|" * len(columns)]
        for name, c in cells.items():
            rows.append(f"| {name} | " + " | ".join(fmt(c[key], digits) for key, _ in columns) + " |")
        return rows

    for variant in ("full", "dale"):
        h2 = st["H2"].get(variant)
        if not h2:
            continue
        L += ["", f"## H2 / E2: interference and facilitation ({variant}; n = {h2['n']})", ""]
        L += cells_table(h2["cells"], [("rt_A", "RT A"), ("rt_B", "RT B"), ("interference_A", "interference A"),
                                       ("interference_B", "interference B"), ("interference_asym", "asymmetry B - A"),
                                       ("facilitation_A", "facilitation A"), ("facilitation_B", "facilitation B")])
        L += ["", "Regression on the grid, y ~ k/10 * log2(rho) (HC3 standard errors):", "",
              "| measure | intercept | k (per 10 epochs) | log2 rho | interaction | R2 |", "|---|---|---|---|---|---|"]
        for m, model in h2["regression"].items():
            L.append(f"| {m} | {term(model, 'Intercept')} | {term(model, 'k_s')} | {term(model, 'lr')} | {term(model, 'k_s:lr')} | "
                     f"{model.get('r2', float('nan')):.3f} |")
        L += ["", "Robustness of the regression of the interference asymmetry:", "",
              "| variant of the measure | k (per 10 epochs) | log2 rho | interaction |", "|---|---|---|---|"]
        for name, model in h2["robustness"].items():
            L.append(f"| {name} | {term(model, 'k_s')} | {term(model, 'lr')} | {term(model, 'k_s:lr')} |")
        nc = h2["norm_control"]
        L += ["", f"Input-norm control (corners of the grid; {nc['n_control']} control and {nc['n_free']} free networks):", "",
              "| measure | model | k (per 10 epochs) | log2 rho | k x control | log2 rho x control |", "|---|---|---|---|---|---|"]
        for m in ("interference_asym", "stroop_asym", "err_interference_asym", "rt_asym"):
            L.append(f"| {m} | all, with interactions | {term(nc[m]['with_interactions'], 'k_s')} | {term(nc[m]['with_interactions'], 'lr')} | "
                     f"{term(nc[m]['with_interactions'], 'k_s:norm')} | {term(nc[m]['with_interactions'], 'lr:norm')} |")
            L.append(f"| {m} | control networks only | {term(nc[m]['control_only'], 'k_s')} | {term(nc[m]['control_only'], 'lr')} | | |")
        if "links" in h2:
            L += ["", "Rank correlations across the networks of the grid:", ""]
            for name, link in h2["links"].items():
                L.append(f"- {name}: Spearman rho = {link['spearman']:.2f}, p {pfmt(link['p'])} (n = {link['n']})")

    for variant in ("full", "dale"):
        me = st.get("mechanism", {}).get(variant)
        if not me:
            continue
        L += ["", f"## Mechanism of the interference: noise-free responses to the two features ({variant}; n = {me['n']})", "",
              "Constant stimuli without any noise, from the state prepared by the cue for 100 or 800 ms. Intrusion = response of the "
              "decision variable to the irrelevant feature alone divided by the response to the relevant feature alone "
              "(\"on B\" = feature A intruding while task B is cued).", "",
              "| condition | interference A, CSI 100 | B, CSI 100 | Wilcoxon p | A, CSI 800 | B, CSI 800 | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
        for name, c in me["cells"].items():
            L.append(f"| {name} | {fmt(c['nf_interference_A_csi100'])} | {fmt(c['nf_interference_B_csi100'])} | "
                     f"{pfmt(c['wilcoxon_nf_interference_csi100_B_vs_A']['p'])} | {fmt(c['nf_interference_A_csi800'])} | "
                     f"{fmt(c['nf_interference_B_csi800'])} | {pfmt(c['wilcoxon_nf_interference_csi800_B_vs_A']['p'])} |")
        L += ["", "| condition | intrusion on A at 50 ms, CSI 100 | on B | Wilcoxon p | on A at 50 ms, CSI 800 | on B | on A at 500 ms, CSI 800 | on B |",
              "|---|---|---|---|---|---|---|---|"]
        for name, c in me["cells"].items():
            L.append(f"| {name} | {fmt(c['nf_intrusion_A_csi100_t50'], 3)} | {fmt(c['nf_intrusion_B_csi100_t50'], 3)} | "
                     f"{pfmt(c['wilcoxon_nf_intrusion_csi100_t50_B_vs_A']['p'])} | {fmt(c['nf_intrusion_A_csi800_t50'], 3)} | "
                     f"{fmt(c['nf_intrusion_B_csi800_t50'], 3)} | {fmt(c['nf_intrusion_A_csi800_t500'], 3)} | {fmt(c['nf_intrusion_B_csi800_t500'], 3)} |")
        L += ["", "| condition | noise-free RT A (neutral, CSI 800) | RT B | initial slope, relevant A (per s) | relevant B |", "|---|---|---|---|---|"]
        for name, c in me["cells"].items():
            scaled = {key: {kk: (vv * 1000 if kk in ("mean", "lo", "hi") and vv is not None else vv) for kk, vv in c[key].items()}
                      for key in ("nf_slope_rel_A", "nf_slope_rel_B")}
            L.append(f"| {name} | {fmt(c['nf_rt_neut_A_csi800'], 1)} | {fmt(c['nf_rt_neut_B_csi800'], 1)} | "
                     f"{fmt(scaled['nf_slope_rel_A'])} | {fmt(scaled['nf_slope_rel_B'])} |")
        if "interference_by_csi_all" in me:
            L += ["", "Behavioural interference (noisy test battery) against the cue-stimulus interval, all networks of the grid: "
                  + ", ".join(f"CSI {ms}: {fmt(s)}" for ms, s in zip(CSI_MS, me["interference_by_csi_all"]))
                  + f"; CSI 100 vs 800: p {pfmt(me['sign_flip_interference_csi100_vs_800']['p'])}.", "",
                  "| condition | interference A at CSI 100 | 200 | 400 | 800 | interference B at CSI 100 | 200 | 400 | 800 |", "|---|---|---|---|---|---|---|---|---|"]
            for name, c in me["cells"].items():
                L.append(f"| {name} | " + " | ".join(fmt(s, 1) for s in c["interference_by_csi"]["A"]) + " | "
                         + " | ".join(fmt(s, 1) for s in c["interference_by_csi"]["B"]) + " |")
        L += ["", "Regression on the learning history, y ~ k/10 * log2(rho) (asymmetry = task B minus task A):", "",
              "| measure | intercept | k (per 10 epochs) | log2 rho | interaction |", "|---|---|---|---|---|"]
        for m, model in me["regression"].items():
            L.append(f"| {m} | {term(model, 'Intercept')} | {term(model, 'k_s')} | {term(model, 'lr')} | {term(model, 'k_s:lr')} |")
        L += ["", "Rank correlations across the networks of the grid:", ""]
        for name, link in me["links"].items():
            L.append(f"- {name}: Spearman rho = {link['spearman']:.2f}, p {pfmt(link['p'])} (n = {link['n']})")

    for variant in ("full", "dale"):
        h3 = st["H3"].get(variant)
        if not h3:
            continue
        L += ["", f"## H3 / E3: selection-vector vs input modulation per feature ({variant}; n = {h3['n']})", "",
              "Medians over networks with bootstrap intervals; Wilcoxon signed-rank tests of feature A against feature B within networks.", "",
              "| condition | sel. fraction A | sel. fraction B | Wilcoxon p | input mod. A | input mod. B | Wilcoxon p | gain rel. A (dv) | gain rel. B (dv) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, c in h3["cells"].items():
            L.append(f"| {name} | {fmt(c['sel_fraction_A'], 3)} | {fmt(c['sel_fraction_B'], 3)} | {pfmt(c['wilcoxon_sel_fraction_A_vs_B']['p'])} | "
                     f"{fmt(c['inp_mod_A'])} | {fmt(c['inp_mod_B'])} | {pfmt(c['wilcoxon_inp_mod_A_vs_B']['p'])} | "
                     f"{fmt(c['gain_rel_dv_A'])} | {fmt(c['gain_rel_dv_B'])} |")
        L += ["", f"All networks with overlearning (k > 0 or rho > 1): selection-vector fraction of A minus that of B "
              f"{fmt(h3['sel_fraction_diff_overlearned'], 3)} (median), "
              f"Wilcoxon p {pfmt(h3['wilcoxon_overlearned_sel_fraction_A_vs_B']['p'])} "
              f"(n = {h3['wilcoxon_overlearned_sel_fraction_A_vs_B']['n']})."
              if h3.get("median_sel_fraction_diff_overlearned") is not None else ""]
        L += ["", "Median regression on the learning history, y ~ k/10 * log2(rho) (`_diff` = feature A minus feature B; "
              "`norm_in_diff` and `edim`: least squares):", "",
              "| measure | intercept | k (per 10 epochs) | log2 rho | interaction |", "|---|---|---|---|---|"]
        for m, model in h3["regression_no_covariate"].items():
            L.append(f"| {m} | {term(model, 'Intercept')} | {term(model, 'k_s')} | {term(model, 'lr')} | {term(model, 'k_s:lr')} |")
        L += ["", "The same with the effective dimension of W_rec (centred) as covariate, y ~ k/10 + log2(rho) + dimension:", "",
              "| measure | intercept | k (per 10 epochs) | log2 rho | effective dimension |", "|---|---|---|---|---|"]
        for m, model in h3["regression"].items():
            L.append(f"| {m} | {term(model, 'Intercept')} | {term(model, 'k_s')} | {term(model, 'lr')} | {term(model, 'edim_c')} |")
        p = h3["perturbation"]
        L += ["", "State perturbations at stimulus onset (change of the decision variable per unit perturbation, no noise):", "",
              f"- along the selection vector {fmt(p['sensitivity_selection'], 3)} (linear prediction {fmt(p['predicted_selection'], 3)}; "
              f"measured / predicted {fmt(p['measured_over_predicted'])}; rank correlation across networks {p['spearman_predicted_measured']:.2f})",
              f"- along the line attractor {fmt(p['sensitivity_line'], 3)}; random directions of the same norm {fmt(p['sensitivity_random'], 3)}",
              f"- selection vs random: ratio {fmt(p['ratio_selection_to_random'], 1)}, Wilcoxon p {pfmt(p['wilcoxon_selection_vs_random']['p'])}; "
              f"selection vs line attractor: Wilcoxon p {pfmt(p['wilcoxon_selection_vs_line']['p'])}",
              "- choice bias P(+ | +a) - P(+ | -a) with noisy stimuli, by amplitude a = 1, 2, 4, 8: selection "
              + ", ".join(fmt(p["choice_bias"]["sel"][a], 3) for a in ("1", "2", "4", "8"))
              + "; random " + ", ".join(fmt(p["choice_bias"]["rand"][a], 3) for a in ("1", "2", "4", "8"))]

    for variant in ("full", "dale"):
        h4 = st["H4"].get(variant)
        if not h4:
            continue
        L += ["", f"## H4 / E4: switch costs ({variant}; n = {h4['n']})", ""]
        L += cells_table(h4["cells"], [("switch_cost_A", "cost into A"), ("switch_cost_B", "cost into B"), ("switch_asym", "asymmetry A - B"),
                                       ("premature_repeat_A", "premature, repeat A (%)"), ("premature_repeat_B", "premature, repeat B (%)")])
        L += ["", "| condition | cost at CSI 100 (A) | (B) | cost at CSI 800 (A) | (B) | CSI effect p | asymmetry p |", "|---|---|---|---|---|---|---|"]
        for name, c in h4["cells"].items():
            L.append(f"| {name} | {fmt(c['cost_csi100']['A'])} | {fmt(c['cost_csi100']['B'])} | {fmt(c['cost_csi800']['A'])} | "
                     f"{fmt(c['cost_csi800']['B'])} | {pfmt(c['sign_flip_csi_effect']['p'])} | {pfmt(c['sign_flip_asym']['p'])} |")
        mm = h4["mixed_model"]
        L += ["", f"Linear mixed-effects model on cell means: {mm.get('formula', '')}", ""]
        if "terms" in mm:
            L += [f"({mm['n_rows']} cells from {mm['n_networks']} networks; {mm['coding']})", "", "| term | estimate [95% CI] | p |", "|---|---|---|"]
            for name, t in mm["terms"].items():
                L.append(f"| {name} | {t['coef']:.2f} [{t['lo']:.2f}, {t['hi']:.2f}] | {pfmt(t['p'])} |")
        else:
            L.append(f"model could not be fitted: {mm.get('error')}")
        L += ["", "Regression of the per-network asymmetry (cost into A - cost into B), y ~ k/10 * log2(rho):", "",
              "| measure | intercept | k (per 10 epochs) | log2 rho | interaction |", "|---|---|---|---|---|"]
        for m, model in {**h4["regression"], **{f"switch_asym ({k})": v for k, v in h4["robustness"].items()}}.items():
            L.append(f"| {m} | {term(model, 'Intercept')} | {term(model, 'k_s')} | {term(model, 'lr')} | {term(model, 'k_s:lr')} |")
        r = h4["relaxation"]
        L += ["", "Time for the state to reach the cued context's slow points (noise-free), and the slowest eigenvalues:", "",
              f"- from the other context's structure: time to 1/e of the initial distance {fmt(r['t_e_switch'], 0)} ms, tail time constant {fmt(r['tau_tail_switch'], 0)} ms",
              f"- from the resting state: time to 1/e {fmt(r['t_e_rest'], 0)} ms, tail time constant {fmt(r['tau_tail_rest'], 0)} ms",
              f"- -tau / Re(lambda) of the slowest mode other than the decision mode: {fmt(r['tau_eig2'], 0)} ms; tail time constant / this value: "
              f"{fmt(r['ratio_tau_tail_switch_to_tau_eig2'])}; rank correlation across networks and contexts "
              f"{r['spearman_tau_tail_switch_vs_tau_eig2']['spearman']:.2f} (p {pfmt(r['spearman_tau_tail_switch_vs_tau_eig2']['p'])})",
              f"- rank correlation between the time to 1/e and the switch cost at the shortest CSI: "
              f"{r['spearman_t_e_switch_vs_cost_short_csi']['spearman']:.2f} (p {pfmt(r['spearman_t_e_switch_vs_cost_short_csi']['p'])})",
              f"- remaining distance at stimulus onset for CSI = 100, 200, 400, 800 ms: after a switch "
              + ", ".join(f"{x:.1f}" for x in r["dist_at_csi_switch"]) + "; from rest " + ", ".join(f"{x:.1f}" for x in r["dist_at_csi_rest"])]

    for variant in ("full", "dale"):
        h5 = st["H5"].get(variant)
        if not h5 or not h5["conditions"]:
            continue
        L += ["", f"## H5 / E5: state resets at cue onset ({variant})", "",
              "Costs are RT differences to repeat trials without intervention (response-cue interval 100 ms), averaged over the four CSIs "
              "unless a CSI is given. Asymmetry = cost for task A - cost for task B.", "",
              "| condition | arm | cost A | cost B | asymmetry (p) | cost at CSI 100 | cost at CSI 800 | CSI effect p |", "|---|---|---|---|---|---|---|---|"]
        for name, c in h5["conditions"].items():
            for key, label in (("cost_none", "no intervention"), ("cost_target", "reset to target structure"),
                               ("cost_neutral", "reset to resting state")):
                d = c[key]
                L.append(f"| {name} | {label} | {fmt(d['A'])} | {fmt(d['B'])} | {fmt(d['asym'])} ({pfmt(d['test_asym']['p'])}) | "
                         f"{d['test_csi100']['mean']:.2f} (p {pfmt(d['test_csi100']['p'])}) | {d['test_csi800']['mean']:.2f} (p {pfmt(d['test_csi800']['p'])}) | "
                         f"{pfmt(d['test_csi_effect']['p'])} |")
        L += ["", "Additive split of the switch cost, RT(switch) - RT(repeat) = inertia + reconfiguration + repetition benefit:", "",
              "| condition | component | A | B | asymmetry (p) | at CSI 100: A | B |", "|---|---|---|---|---|---|---|"]
        for name, c in h5["conditions"].items():
            for key, label in (("inertia", "inertia: switch - reset to rest"), ("reconfiguration", "reconfiguration: reset to rest - reset to target"),
                               ("repetition_benefit", "repetition benefit: reset to target - repeat")):
                d = c[key]
                L.append(f"| {name} | {label} | {fmt(d['A'])} | {fmt(d['B'])} | {fmt(d['asym'])} ({pfmt(d['test_asym']['p'])}) | "
                         f"{fmt(d['A_csi100'])} | {fmt(d['B_csi100'])} |")
        L += ["", "Response times at the shortest CSI (100 ms) in every arm, and on repeat trials after longer response-cue intervals "
              "(share of premature responses in parentheses):", "",
              "| condition | task | repeat, RCI 100 | repeat, RCI 600 | repeat, RCI 2000 | repeat, RCI 5000 | switch | switch, reset to target | switch, reset to rest |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, c in h5["conditions"].items():
            for task in "AB":
                cells_ = [f"{fmt(c['csi100']['repeat'][f'rt_{task}'], 1)} ({c['csi100']['repeat'][f'premature_{task}']:.2f})"]
                cells_ += [f"{fmt(c['long_rsi']['csi100'][f'rci{ms}'][f'rt_{task}'], 1)} ({c['long_rsi']['csi100'][f'rci{ms}'][f'premature_{task}']:.2f})"
                           for ms in LONG_RCI_MS[1:]]
                cells_ += [f"{fmt(c['csi100'][arm][f'rt_{task}'], 1)} ({c['csi100'][arm][f'premature_{task}']:.2f})" for arm in ARMS[1:]]
                L.append(f"| {name} | {task} | " + " | ".join(cells_) + " |")
        L += ["", "Asymmetries (A - B) without the exclusion of premature responses:", "",
              "| condition | no intervention | reset to target | reset to rest | inertia | reconfiguration |", "|---|---|---|---|---|---|"]
        for name, c in h5["conditions"].items():
            L.append(f"| {name} | " + " | ".join(f"{fmt(c['raw'][key]['asym'])} ({pfmt(c['raw'][key]['test_asym']['p'])})"
                                                for key in ("cost_none", "cost_target", "cost_neutral", "inertia", "reconfiguration")) + " |")
        L += ["", "Repeat trials after a long response-cue interval (5000 ms vs 100 ms; CSI 100 ms):", "",
              "| condition | cost A | cost B | asymmetry A - B (p) | asymmetry 5000 vs 600 ms (p) |", "|---|---|---|---|---|"]
        for name, c in h5["conditions"].items():
            d, e = c["long_rsi"]["5000_vs_100"], c["long_rsi"]["5000_vs_600"]
            L.append(f"| {name} | {fmt(d['A_csi100'])} | {fmt(d['B_csi100'])} | {fmt(d['asym_csi100'])} ({pfmt(d['test_asym_csi100']['p'])}) | "
                     f"{fmt(e['asym_csi100'])} ({pfmt(e['test_asym_csi100']['p'])}) |")
        if "regression" in h5:
            L += ["", "Dependence of the asymmetries on the learning history, y ~ k/10 * log2(rho):", "",
                  "| asymmetry of | intercept | k (per 10 epochs) | log2 rho | interaction |", "|---|---|---|---|---|"]
            for m, model in h5["regression"].items():
                L.append(f"| {m} | {term(model, 'Intercept')} | {term(model, 'k_s')} | {term(model, 'lr')} | {term(model, 'k_s:lr')} |")

    h6 = st.get("H6", {})
    if h6.get("dale_vs_full"):
        L += ["", "## H6 / E6: Dale's law and the second optimiser", "",
              "Both variants in one regression, y ~ (k/10 + log2 rho) * dale:", "",
              "| measure | k (per 10 epochs) | log2 rho | dale | k x dale | log2 rho x dale |", "|---|---|---|---|---|---|"]
        for m, model in h6["dale_vs_full"].items():
            L.append(f"| {m} | {term(model, 'k_s')} | {term(model, 'lr')} | {term(model, 'dale')} | {term(model, 'k_s:dale')} | {term(model, 'lr:dale')} |")
    for variant, entry in h6.get("optimizer", {}).items():
        L += ["", f"SGD with momentum vs Adam ({variant}; baseline k = 0, rho = 1 and overlearned k = {K_LEVELS[-1]}, rho = {RHO_LEVELS[-1]}; "
              f"n = {entry['n']}):", "",
              "| measure | Adam baseline | Adam overlearned | SGD baseline | SGD overlearned | overlearning effect Adam (p) | SGD (p) |", "|---|---|---|---|---|---|---|"]
        for m in H6_MEASURES + H6_DYNAMICS:
            e = entry[m]
            L.append(f"| {m} | {fmt(e['adam_base'], 3)} | {fmt(e['adam_over'], 3)} | {fmt(e['sgd_base'], 3)} | {fmt(e['sgd_over'], 3)} | "
                     f"{e['overlearning_effect_adam']['diff']:.3f} ({pfmt(e['overlearning_effect_adam']['p'])}) | "
                     f"{e['overlearning_effect_sgd']['diff']:.3f} ({pfmt(e['overlearning_effect_sgd']['p'])}) |")

    la = st.get("line_attractor_all")
    if la:
        L += ["", "## Line-attractor criteria in every condition", "",
              "| condition | n | criteria met | median abs(lambda tau) | max abs(lambda tau) | sel. fraction A | sel. fraction B |", "|---|---|---|---|---|---|---|"]
        for name, c in la.items():
            L.append(f"| {name} | {c['n']} | {100 * c['frac_line_attractor']:.0f}% | {c['median_abs_lambda']:.3f} | {c['max_abs_lambda']:.3f} | "
                     f"{fmt(c['sel_fraction_A'], 3)} | {fmt(c['sel_fraction_B'], 3)} |")

    snap = st.get("snapshots")
    if snap:
        L += ["", "## Behaviour during joint training (snapshots)", "",
              "Epoch 0 is the end of the pre-training on task A. Measures use the networks above the accuracy criterion at that moment.", "",
              "| condition | joint epoch | n above criterion | accuracy A | accuracy B | RT B - RT A | interference asymmetry | switch asymmetry |",
              "|---|---|---|---|---|---|---|---|"]
        for name, epochs in snap.items():
            for epoch, c in sorted(epochs.items(), key=lambda kv: int(kv[0])):
                L.append(f"| {name} | {epoch} | {c['n_above_criterion']}/{c['n']} | {fmt(c['acc_A'], 3)} | {fmt(c['acc_B'], 3)} | {fmt(c['rt_asym'])} | "
                         f"{fmt(c['interference_asym'])} | {fmt(c['switch_asym'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = ROOT / args.results
    df = add_derived(pd.read_csv(results / "networks.csv"))
    arrays = dict(np.load(results / "arrays.npz"))

    st = {"exclusions": exclusions(df), "H1": {}, "H2": {}, "mechanism": {}, "H3": {}, "H4": {}, "H5": {}}
    for variant in ("full", "dale"):
        st["H1"][variant] = h1_replication(df, variant)
        st["H2"][variant] = h2_interference(df, variant)
        st["mechanism"][variant] = mechanism(df, arrays, variant)
        st["H3"][variant] = h3_selection(df, variant)
        st["H4"][variant] = h4_switch(df, arrays, variant)
        st["H5"][variant] = h5_interventions(df, arrays, variant)
    st["V"] = v_rank_one(df)
    st["H6"] = h6_robustness(df)
    st["line_attractor_all"] = line_attractor_all(df)
    st["snapshots"] = snapshots(results)
    e0 = results / "e0_validation.json"
    if e0.exists():
        st["E0"] = json.loads(e0.read_text())

    (results / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    (results / "report.md").write_text(report(st))
    print(f"written {results / 'stats.json'} and {results / 'report.md'}")


if __name__ == "__main__":
    main()
