"""Regenerate every figure of the study from the tables in results/.

    python scripts/make_figures.py                 # results/ -> figures/fig01_... fig11_... (PNG and PDF)
    python scripts/make_figures.py --only 5 9      # selected figures

Points and lines show means over networks with 95% bootstrap confidence intervals, unless a panel says
"median" (used for ratios such as the selection-vector fraction, which have heavy tails). Task A is the
overlearned task (k epochs of pre-training, rho times as many trials as task B during joint training).
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "runs" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stroop_rnn.config import K_LEVELS, MAIN_THRESHOLD, RHO_LEVELS, RT_THRESHOLDS
from stroop_rnn.dynamics import VALLEY_D
from stroop_rnn.experiments import PSY_LEVELS
from stroop_rnn.stats import bootstrap_ci
from stroop_rnn.validation import WongWang

ROOT = Path(__file__).resolve().parents[1]
TH = MAIN_THRESHOLD
TI = RT_THRESHOLDS.index(TH)
CSI_MS = np.array([100, 200, 400, 800])
RCI_MS = np.array([100, 200, 400, 600])
LONG_RCI_MS = np.array([100, 600, 2000, 5000])
C_A, C_B = "#1b6ca8", "#d1495b"                    # task A (overlearned), task B
C_K = {0: "#9aa5b1", 5: "#52606d", 10: "#111827"}   # pre-training epochs
C_ARM = {"repeat": "#6b7280", "switch": "#111827", "switch_to_target": "#2a9d8f", "switch_to_neutral": "#e9a03b"}
ARM_LABEL = {"repeat": "repeat", "switch": "switch", "switch_to_target": "switch, reset to target",
             "switch_to_neutral": "switch, reset to rest"}
ARMS = tuple(C_ARM)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
    "savefig.dpi": 200, "legend.frameon": False, "lines.linewidth": 1.5, "axes.titleweight": "bold",
    "legend.handlelength": 3.2,                      # long enough to tell dashed from solid lines
})


# ------------------------------------------------------------------------------- helpers
class Data:
    def __init__(self, results: Path):
        self.dir = results
        self.df = pd.read_csv(results / "networks.csv")
        self.arr = np.load(results / "arrays.npz")
        self.hist = np.load(results / "training_history.npz")
        self.e0 = np.load(results / "e0_validation.npz") if (results / "e0_validation.npz").exists() else None
        self.example = np.load(results / "example_network.npz") if (results / "example_network.npz").exists() else None
        snap = results / "snapshots.csv"
        self.snap = pd.read_csv(snap) if snap.exists() else None

    def idx(self, variant="full", k=None, rho=None, norm=False, optimizer="adam") -> np.ndarray:
        """Row numbers (= positions in the arrays) of the included networks of a set of conditions."""
        d = self.df
        m = (d["variant"] == variant) & (d["norm_control"] == norm) & (d["optimizer"] == optimizer) & d["included"]
        if k is not None:
            m &= d["k"] == k
        if rho is not None:
            m &= d["rho"] == rho
        return np.flatnonzero(m.to_numpy())

    def col(self, name: str, rows: np.ndarray, theta: float = TH) -> np.ndarray:
        key = name if name in self.df else f"{name}@{theta}"
        return self.df[key].to_numpy(dtype=float)[rows]


def ci(values, stat=np.mean):
    """Estimate and 95% bootstrap interval over networks (NaN-safe)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    est, lo, hi = bootstrap_ci(v, stat=stat, n_boot=2000)
    return float(est), float(lo), float(hi)


def ci_curve(values: np.ndarray):
    """Mean and bootstrap interval for every column of values [networks, points]."""
    out = np.array([ci(values[:, j]) for j in range(values.shape[1])])
    return out[:, 0], out[:, 1], out[:, 2]


def errline(ax, x, groups, stat=np.mean, **kw):
    """Line through the estimates of several groups of networks, with confidence intervals as error bars."""
    est = np.array([ci(g, stat) for g in groups])
    ax.errorbar(x, est[:, 0], yerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]], capsize=2, marker="o", ms=3.5, **kw)


def band(ax, x, values, color, label=None, ls="-", alpha=0.18):
    mean, lo, hi = ci_curve(values)
    ax.plot(x, mean, color=color, ls=ls, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)


def strip(ax, x, values, color, width=0.28, seed=0, **kw):
    """Individual networks as jittered points plus the median as a bar."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    ax.scatter(x + rng.uniform(-width / 2, width / 2, len(v)), v, s=9, color=color, alpha=0.6, lw=0, **kw)
    ax.plot([x - width, x + width], [np.median(v)] * 2, color="k", lw=1.5)


def grid_lines(ax, data: Data, variant: str, getter, by_task: bool, stat=np.mean, ls_variant="-", color=None, label_prefix=""):
    """One line per number of pre-training epochs against the task ratio rho.

    getter(rows) -> values per network, or (values for A, values for B) if by_task.
    """
    x = np.log2(RHO_LEVELS)
    for k in K_LEVELS:
        rows = [data.idx(variant, k, rho) for rho in RHO_LEVELS]
        if any(len(r) == 0 for r in rows):
            continue
        off = (K_LEVELS.index(k) - 1) * 0.035
        if by_task:
            for t, (c, name) in enumerate(((C_A, "A"), (C_B, "B"))):
                errline(ax, x + off, [getter(r)[t] for r in rows], stat, color=c, alpha=0.35 + 0.325 * K_LEVELS.index(k),
                        ls=(":", "--", "-")[K_LEVELS.index(k)], label=f"{label_prefix}task {name}, k = {k}")
        else:
            errline(ax, x + off, [getter(r) for r in rows], stat, color=color or C_K[k], ls=ls_variant,
                    label=f"{label_prefix}k = {k}")
    ax.set_xticks(x, [str(r) for r in RHO_LEVELS])
    ax.set_xlabel("task ratio rho (A : B trials)")


def panel_labels(axes, width: int = 32):
    """Letter every panel and wrap titles that are wider than their panel."""
    for i, ax in enumerate(np.ravel(axes)):
        lines = ax.get_title().split("\n")
        if max(len(line) for line in lines) > width + 2:
            ax.set_title("\n".join(textwrap.fill(line, width) for line in lines))
        ax.text(-0.17, 1.06, "abcdefghijkl"[i], transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"written {out / name}.png", flush=True)


# ----------------------------------------------------------------- Fig. 1: validation (E0, V)
def fig01_validation(data: Data, out: Path) -> None:
    e0, df = data.e0, data.df
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    ax = axes[0, 0]
    ax.scatter(e0["lin_l_true"], e0["lin_l_found"], s=14, color=C_A)
    lim = np.abs(e0["lin_l_true"]).max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.8, ls=":")
    ax.set(xlabel="true selection vector (component)", ylabel="recovered", title="Linear line attractor: selection vector")

    ax = axes[0, 1]
    ax.scatter(e0["lin_predicted"], e0["lin_moved"], s=18, color=C_A, zorder=3)
    lim = np.abs(e0["lin_predicted"]).max() * 2
    ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.8, ls=":")
    ax.set_xscale("symlog", linthresh=1.0)             # the ten systems differ by orders of magnitude
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set(xlabel="predicted from the effective gain l . b", ylabel="measured displacement along the line",
           title="Linear line attractor: integration", xlim=(-lim, lim), ylim=(-lim, lim))

    ax = axes[0, 2]
    model = WongWang(mu0=30.0, coh=0.0)
    s = np.linspace(0.0, 1.0, 401)
    s1, s2 = np.meshgrid(s, s, indexing="ij")
    f = model.field(np.stack([s1, s2], axis=-1))
    ax.contour(s1, s2, f[..., 0], levels=[0.0], colors=[C_A], linewidths=1.2)
    ax.contour(s1, s2, f[..., 1], levels=[0.0], colors=[C_B], linewidths=1.2)
    found, stable = e0["ww_found_0"], e0["ww_stable_0"]
    ax.scatter(found[stable, 0], found[stable, 1], s=45, color="k", zorder=3, label="stable (search)")
    ax.scatter(found[~stable, 0], found[~stable, 1], s=45, facecolor="w", edgecolor="k", zorder=3, label="saddle (search)")
    ax.plot([], [], color=C_A, label="nullcline dS1/dt = 0")
    ax.plot([], [], color=C_B, label="nullcline dS2/dt = 0")
    ax.set(xlabel="S1", ylabel="S2", title="Wong & Wang (2006): fixed points", xlim=(0, 0.8), ylim=(0, 0.8))
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    groups = [("full", data.idx("full", 0, 1), C_K[10]), ("Dale", data.idx("dale", 0, 1), C_K[5]),
              ("rank-one", data.idx("rank1"), C_B)]
    for i, (name, rows, c) in enumerate(groups):
        strip(ax, i, np.abs(data.col("cos_l", rows)), c)
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set(ylabel="|cos| between the selection vectors\nof the two contexts", title="V: selection vector cannot rotate", ylim=(-0.05, 1.08))

    ax = axes[1, 1]
    for i, (name, rows, c) in enumerate(groups):
        both = np.concatenate([data.col("sel_fraction_unit_A", rows), data.col("sel_fraction_unit_B", rows)])
        if name == "rank-one" and "r1_sel_fraction_aligned" in e0:
            both = e0["r1_sel_fraction_aligned"][:, :, 0].ravel()      # the sign of a selection vector is a convention
        strip(ax, i, np.clip(both, -0.5, 1.5), c)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.text(0.98, 0.5, "rank-one: both selection\nvectors oriented the same way", transform=ax.transAxes, fontsize=6.5, ha="right")
    ax.set(ylabel="selection-vector fraction\n(unit-norm selection vectors)", title="V: share of selection-vector modulation")

    ax = axes[1, 2]
    for j, (name, c) in enumerate((("bias", "#6b7280"), ("feature 1", C_A), ("feature 2", C_B))):
        ax.scatter(e0["rec_true"][:, j], e0["rec_fitted"][:, j], s=7, color=c, alpha=0.6, lw=0, label=name)
    lim = (e0["rec_true"].min() - 0.5, e0["rec_true"].max() + 0.5)
    ax.plot(lim, lim, color="k", lw=0.8, ls=":")
    ax.set(xlabel="true weight of the simulated agent", ylabel="fitted weight", title="Logistic regression: parameter recovery")
    ax.legend()
    panel_labels(axes)
    fig.tight_layout()
    save(fig, out, "fig01_validation")


# ------------------------------------------------------------ Fig. 2: task and training
def fig02_task_training(data: Data, out: Path) -> None:
    ex, hist = data.example, data.hist
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.5])
    ax1, ax2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    t = np.arange(len(ex["seq_x"])) * float(ex["dt"]) / 1000.0
    x, y, target = ex["seq_x"], ex["seq_y"], ex["seq_target"]
    for ax in (ax1, ax2):
        for i, (on, off) in enumerate(zip(ex["seq_cue_on"], ex["seq_cue_off"])):
            c = C_A if ex["seq_context"][i] == 0 else C_B
            ax.axvspan(t[on], t[min(off, len(t) - 1)], color=c, alpha=0.10, lw=0)
    ax1.plot(t, x[:, 0], color=C_A, lw=0.7, label="feature A (noisy)")
    ax1.plot(t, x[:, 1], color=C_B, lw=0.7, label="feature B (noisy)")
    for i, on in enumerate(ex["seq_stim_on"]):
        cong = ("congruent", "neutral", "incongruent")[int(ex["seq_congruency"][i])]
        sw = {-1: "first", 0: "repeat", 1: "switch"}[int(ex["seq_switch"][i])]
        ax1.text(t[ex["seq_cue_on"][i]], ax1.get_ylim()[1] * 1.02, f"cue {'AB'[int(ex['seq_context'][i])]}: {sw}\n{cong}",
                 fontsize=7, va="bottom")
    ax1.set_ylabel("stimulus input")
    ax1.set_title("One sequence of six trials without reset (shading = cue on)", pad=26)
    ax1.legend(loc="lower right", ncol=2)
    ax2.plot(t, y[:, 0] - y[:, 1], color="k", label="decision variable y1 - y2")
    ax2.plot(t, target[:, 0] - target[:, 1], color="#2a9d8f", ls="--", lw=1.1, label="target")
    for th in (TH, -TH):
        ax2.axhline(th, color="#9aa5b1", lw=0.7, ls=":")
    ax2.set(xlabel="time (s)", ylabel="output", xlim=ax1.get_xlim())
    ax2.legend(loc="lower right", ncol=2)

    conds = [("full_k0_rho1", C_K[0], "k = 0, rho = 1"), ("full_k10_rho1", C_K[5], "k = 10, rho = 1"),
             ("full_k10_rho4", C_K[10], "k = 10, rho = 4")]
    ax = fig.add_subplot(gs[2, 0])
    for name, c, label in conds:
        if f"{name}/loss" in hist:
            ax.plot(hist[f"{name}/step"], np.nanmean(hist[f"{name}/loss"], axis=1), color=c, label=label)
    ax.set(yscale="log", xlabel="training step", ylabel="loss", title="Training loss")
    ax.legend()
    ax = fig.add_subplot(gs[2, 1])
    for name, c, label in conds:
        if f"{name}/acc_A" in hist:
            ax.plot(hist[f"{name}/step"], np.nanmean(hist[f"{name}/acc_A"], axis=1), color=c, label=f"task A, {label}")
            ax.plot(hist[f"{name}/step"], np.nanmean(hist[f"{name}/acc_B"], axis=1), color=c, ls="--", label="task B")
    ax.axhline(0.9, color="#9aa5b1", lw=0.7, ls=":")
    ax.set(xlabel="training step", ylabel="accuracy on training batches", title="Accuracy (dashed: task B)", ylim=(0.4, 1.01))
    ax.legend(fontsize=6.5)
    ax = fig.add_subplot(gs[2, 2])
    for name, c, label in (("full_k10_rho4", C_K[10], "k = 10, rho = 4"), ("full_k10_rho4_norm", "#2a9d8f", "same, norm control")):
        if f"{name}/norm_in_A" in hist:
            ax.plot(hist[f"{name}/step"], np.nanmean(hist[f"{name}/norm_in_A"], axis=1), color=c, label=f"feature A, {label}")
            ax.plot(hist[f"{name}/step"], np.nanmean(hist[f"{name}/norm_in_B"], axis=1), color=c, ls="--", label="feature B")
    ax.set(xlabel="training step", ylabel="norm of the input weights", title="Input weights (dashed: feature B)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "fig02_task_training")


# ---------------------------------------------------- Fig. 3: replication (E1 / H1)
def fig03_replication(data: Data, out: Path) -> None:
    arr = data.arr
    rows = data.idx("full", 0, 1)
    levels = np.asarray(PSY_LEVELS)
    p = arr["psy_p_plus"][rows]                                  # [n, context, value of A, value of B]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for c, (ax, cue) in enumerate(zip(axes[0, :2], "AB")):
        band(ax, levels, p[:, c].mean(axis=2), C_A, "against feature A")
        band(ax, levels, p[:, c].mean(axis=1), C_B, "against feature B")
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set(xlabel="signed feature value (units of s)", ylabel="P(choice +)", title=f"Choices when task {cue} is cued", ylim=(-0.02, 1.02))
        ax.legend(loc="upper left")

    ax = axes[0, 2]
    for i, (name, variant, c) in enumerate((("full", "full", C_K[10]), ("Dale", "dale", C_K[5]), ("rank-one", "rank1", "#7c3aed"))):
        r = data.idx(variant, 0, 1)
        rel = (data.col("psy_bA_ctxA", r) + data.col("psy_bB_ctxB", r)) / 2
        irr = (data.col("psy_bB_ctxA", r) + data.col("psy_bA_ctxB", r)) / 2
        strip(ax, i - 0.2, rel, c, width=0.14)
        strip(ax, i + 0.2, irr, c, width=0.14, marker="s")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xticks(range(3), ["full", "Dale", "rank-one"])
    ax.set(ylabel="logistic weight", title="Weight of the relevant (left)\nand irrelevant (right) feature")

    ax = axes[1, 0]
    ev = arr["eigvals_ref"][rows][:, 0]                          # context A, [n, N]
    ax.scatter(ev.real.ravel(), ev.imag.ravel(), s=3, color="#9aa5b1", alpha=0.35, lw=0)
    top = ev.real.argmax(axis=1)
    ax.scatter(ev.real[np.arange(len(ev)), top], ev.imag[np.arange(len(ev)), top], s=16, color=C_B, zorder=3, label="decision mode")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.axvspan(-0.1, 0.1, color=C_B, alpha=0.08, lw=0)
    ax.set(xlabel="Re(lambda) tau", ylabel="Im(lambda) tau", title="Spectrum at the slow point (context A)", xlim=(-2.2, 0.3))
    ax.legend(loc="upper left")

    d = np.asarray(VALLEY_D)
    for ax, key, ylabel, title in ((axes[1, 1], "valley_lam", "eigenvalue of the decision\nmode (lambda tau)", "Along the line of slow points"),
                                   (axes[1, 2], "valley_speed", "speed tau |F(h)|", "Speed along the line of slow points")):
        v = arr[key][rows]                                       # [n, context, 9]
        for c, col in enumerate((C_A, C_B)):
            for trace in v[:, c]:
                ax.plot(d, trace, color=col, lw=0.4, alpha=0.3)
            ax.plot(d, np.nanmedian(v[:, c], axis=0), color=col, lw=2, label=f"context {'AB'[c]} (median)")
        ax.set(xlabel="decision variable y1 - y2 of the slow point", ylabel=ylabel, title=title)
        ax.legend()
    axes[1, 1].axhspan(-0.1, 0.1, color="#9aa5b1", alpha=0.15, lw=0)
    axes[1, 1].set_ylim(-0.6, 0.3)
    axes[1, 2].set_ylim(0, 0.6)
    panel_labels(axes)
    fig.suptitle("Baseline networks (k = 0, rho = 1): context-dependent choices and an approximate line attractor", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig03_replication")


# ----------------------------------------- Fig. 4: state space and selection mechanism (E1, V)
def fig04_state_space(data: Data, out: Path) -> None:
    ex = data.example
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    cmap = plt.get_cmap("coolwarm")
    for c, cue in enumerate("AB"):
        t0, t1 = int(ex[f"stim_on_{cue}"]), int(ex[f"stim_off_{cue}"])
        line = ex[f"proj_line_{cue}"].reshape(-1, 6, 6)          # [time, value of A, value of B]
        for j, feat in enumerate("AB"):
            ax = axes[c, j]
            inp = ex[f"proj_in{feat}_{cue}"].reshape(-1, 6, 6)
            other = 2 - j                                          # average over the other feature
            xs, ys = line.mean(axis=other), inp.mean(axis=other)
            for i in range(6):
                ax.plot(xs[t0:t1, i], ys[t0:t1, i], color=cmap(i / 5), lw=1.4)
                ax.scatter(xs[t1 - 1, i], ys[t1 - 1, i], s=14, color=cmap(i / 5), zorder=3)
            ax.scatter(ex[f"valley_line_{cue}"], np.zeros(9), marker="x", s=18, color="k", lw=0.8, zorder=2)
            relevant = feat == cue
            ax.set(xlabel="position along the line attractor (l . dr)", ylabel=f"input axis of feature {feat}",
                   title=f"Task {cue} cued, sorted by feature {feat} ({'relevant' if relevant else 'irrelevant'})")
    axes[0, 0].text(0.02, 0.97, "colour: signed feature value\ncrosses: slow points", transform=axes[0, 0].transAxes, fontsize=7, va="top")

    ax = axes[0, 2]
    rows = data.idx("full", 0, 1)
    for i, (key, label) in enumerate((("cos_l", "selection\nvectors"), ("cos_r", "line\nattractors"), ("cos_b_A", "input dir.\nfeature A"),
                                     ("cos_b_B", "input dir.\nfeature B"))):
        strip(ax, i, data.col(key, rows), C_K[10])
        ax.text(i, 1.12, label, ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xticks([])
    ax.set(ylabel="cosine between the two contexts", title="What changes with context\n\n", ylim=(-0.5, 1.1))

    ax = axes[1, 2]
    pos = 0
    ticks = []
    for name, variant, c in (("full", "full", C_K[10]), ("Dale", "dale", C_K[5]), ("rank-one", "rank1", "#7c3aed")):
        r = data.idx(variant, 0, 1)
        both = np.concatenate([data.col("sel_fraction_A", r), data.col("sel_fraction_B", r)])
        if variant == "rank1" and data.e0 is not None and "r1_sel_fraction_aligned" in data.e0:
            both = data.e0["r1_sel_fraction_aligned"][:, :, 1].ravel()  # both selection vectors oriented the same way
        strip(ax, pos, np.clip(both, -0.3, 1.3), c)
        ticks.append(name)
        pos += 1
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_xticks(range(3), ticks)
    ax.set(ylabel="selection-vector fraction", title="Selection-vector vs input modulation (Eq. 5)")
    panel_labels(axes)
    fig.suptitle(f"State trajectories of one baseline network ({ex['condition']}, network {int(ex['net'])}) during the stimulus, "
                 "and the decomposition of the context effect", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig04_state_space")


def bars(ax, x, groups, color, width=0.25, label=None, stat=np.mean, **kw):
    est = np.array([ci(g, stat) for g in groups])
    ax.bar(x, est[:, 0], width=width, color=color, label=label, **kw)
    ax.errorbar(x, est[:, 0], yerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]], fmt="none", ecolor="k", elinewidth=0.8, capsize=1.5)


# ------------------------------------------- Fig. 5: interference and facilitation (E2 / H2)
def fig05_interference(data: Data, out: Path) -> None:
    col = data.col
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    grid_lines(axes[0, 0], data, "full", lambda r: (col("rt_A", r), col("rt_B", r)), True)
    axes[0, 0].set(ylabel="response time (ms)", title="Response time")
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    grid_lines(axes[0, 1], data, "full", lambda r: (col("interference_A", r), col("interference_B", r)), True)
    axes[0, 1].set(ylabel="RT incongruent - RT neutral (ms)", title="Interference")
    grid_lines(axes[0, 2], data, "full", lambda r: col("interference_asym", r), False)
    axes[0, 2].axhline(0, color="k", lw=0.5, ls=":")
    axes[0, 2].set(ylabel="interference B - interference A (ms)", title="Asymmetry of the interference")
    axes[0, 2].legend()
    grid_lines(axes[1, 0], data, "full", lambda r: (col("facilitation_A", r), col("facilitation_B", r)), True)
    axes[1, 0].set(ylabel="RT neutral - RT congruent (ms)", title="Facilitation")

    ax = axes[1, 1]
    corners = [(0, 1), (0, 4), (10, 1), (10, 4)]
    x = np.arange(len(corners))
    bars(ax, x - 0.15, [col("interference_asym", data.idx("full", k, rho)) for k, rho in corners], C_K[10], 0.28, "free input weights")
    bars(ax, x + 0.15, [col("interference_asym", data.idx("full", k, rho, norm=True)) for k, rho in corners], "#2a9d8f", 0.28,
         "input norms equalised")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, [f"k = {k}\nrho = {rho}" for k, rho in corners])
    ax.set(ylabel="interference B - interference A (ms)", title="Input-norm control")
    ax.legend()

    ax = axes[1, 2]
    xr = np.log2(RHO_LEVELS)
    for i, theta in enumerate(RT_THRESHOLDS):
        errline(ax, xr + (i - 1) * 0.035, [col("interference_asym", data.idx("full", 10, rho), theta) for rho in RHO_LEVELS],
                color=("#9aa5b1", "#111827", "#7c3aed")[i], label=f"threshold {theta}")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xticks(xr, [str(r) for r in RHO_LEVELS])
    ax.set(xlabel="task ratio rho (A : B trials)", ylabel="interference B - interference A (ms)", title="Response threshold (k = 10)")
    ax.legend()
    panel_labels(axes)
    fig.suptitle("E2: the weaker task B suffers more interference the more task A was practised", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig05_interference")


# ----------------------------------------------- Fig. 6: where the interference comes from
def fig06_mechanism(data: Data, out: Path) -> None:
    arr, col = data.arr, data.col
    base, over = data.idx("full", 0, 1), data.idx("full", 10, 4)
    rel, irr = arr["step_resp_rel"], arr["step_resp_irr"]         # [n, task, CSI (100, 800), time]
    t = (np.arange(rel.shape[-1]) + 1) * 10.0
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    ax = axes[0, 0]
    for ci_, (c, label) in enumerate((("#111827", "cue 100 ms before"), ("#9aa5b1", "cue 800 ms before"))):
        band(ax, t, rel[base][:, :, ci_].mean(axis=1), c, f"relevant feature, {label}")
        band(ax, t, irr[base][:, :, ci_].mean(axis=1), c, f"irrelevant feature, {label}", ls="--")
    ax.axhline(TH, color="#9aa5b1", lw=0.6, ls=":")
    ax.set(xlabel="time from stimulus onset (ms)", ylabel="response of the decision variable", title="Baseline, no noise (tasks pooled)")
    ax.legend(fontsize=6.5)
    ax = axes[0, 1]
    for task, (c, name) in enumerate(((C_A, "A"), (C_B, "B"))):
        band(ax, t, rel[over][:, task, 0], c, f"task {name}: relevant feature")
        band(ax, t, irr[over][:, task, 0], c, f"task {name}: irrelevant feature", ls="--")
    ax.axhline(TH, color="#9aa5b1", lw=0.6, ls=":")
    ax.set(xlabel="time from stimulus onset (ms)", ylabel="response of the decision variable",
           title="k = 10, rho = 4, cue 100 ms before")
    ax.legend(fontsize=6.5)

    ax = axes[0, 2]
    st = arr["stroop_rt"][:, TI]                                  # [n, task, congruency, CSI]
    interference = st[:, :, 2] - st[:, :, 1]
    for rows, ls, label in ((base, "--", "k = 0, rho = 1"), (over, "-", "k = 10, rho = 4")):
        for task, (c, name) in enumerate(((C_A, "A"), (C_B, "B"))):
            errline(ax, CSI_MS, [interference[rows][:, task, j] for j in range(4)], color=c, ls=ls, label=f"task {name}, {label}")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set(xscale="log", xlabel="cue-stimulus interval (ms)", ylabel="interference (ms)", title="Interference needs a short preparation time")
    ax.set_xticks(CSI_MS, [str(v) for v in CSI_MS])
    ax.minorticks_off()
    ax.legend(fontsize=6.5)

    grid_lines(axes[1, 0], data, "full",
               lambda r: (col("nf_irr_A_csi100_t50", r) / col("nf_rel_A_csi100_t50", r), col("nf_irr_B_csi100_t50", r) / col("nf_rel_B_csi100_t50", r)), True)
    axes[1, 0].set(ylabel="irrelevant / relevant response at 50 ms", title="Early intrusion of the irrelevant feature\n(cue 100 ms before)")
    axes[1, 0].legend(fontsize=6.5, ncol=2)

    ax = axes[1, 1]
    rows = data.idx("full")
    xv = col("nf_interference_B_csi100", rows) - col("nf_interference_A_csi100", rows)
    sc = ax.scatter(xv, col("interference_asym", rows), c=np.log2(data.df["rho"].to_numpy()[rows]), cmap="viridis", s=10, alpha=0.8, lw=0)
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["rho 1", "2", "4"])
    ax.set(xlabel="noise-free interference, B - A (ms)", ylabel="behavioural interference, B - A (ms)",
           title="Noise-free responses predict behaviour")

    grid_lines(axes[1, 2], data, "full", lambda r: (1000 * col("nf_slope_rel_A", r), 1000 * col("nf_slope_rel_B", r)), True)
    axes[1, 2].set(ylabel="initial slope of the decision variable (1/s)", title="Drive by the relevant feature")
    panel_labels(axes)
    fig.suptitle("Mechanism: the irrelevant feature leaks into the decision while the context is still being set up", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig06_mechanism")


# ------------------------------------- Fig. 7: selection-vector vs input modulation (E3 / H3)
def fig07_decomposition(data: Data, out: Path) -> None:
    col, df = data.col, data.df
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    grid_lines(axes[0, 0], data, "full", lambda r: (col("sel_fraction_A", r), col("sel_fraction_B", r)), True, stat=np.median)
    axes[0, 0].set(ylabel="selection-vector fraction (median)", title="Share of selection-vector modulation", ylim=(0.68, 1.05))
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    grid_lines(axes[0, 1], data, "full", lambda r: (col("inp_mod_A", r), col("inp_mod_B", r)), True, stat=np.median)
    axes[0, 1].set(ylabel="input modulation of the gain (median)", title="Input modulation per feature")

    ax = axes[0, 2]
    rows = data.idx("full")
    a, b = np.clip(col("sel_fraction_A", rows), 0.3, 1.3), np.clip(col("sel_fraction_B", rows), 0.3, 1.3)
    sc = ax.scatter(b, a, c=np.log2(df["rho"].to_numpy()[rows]), cmap="viridis", s=10, alpha=0.8, lw=0)
    ax.plot([0.3, 1.3], [0.3, 1.3], color="k", lw=0.6, ls=":")
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["rho 1", "2", "4"])
    ax.text(0.03, 0.97, "values beyond the axis range\nare drawn at its edge", transform=ax.transAxes, fontsize=6.5, va="top")
    ax.set(xlabel="feature B", ylabel="feature A (overlearned)", title="Selection-vector fraction: every network of the grid")

    ax = axes[1, 0]
    labels = ["selection\nvector", "linear\nprediction", "line\nattractor", "random"]
    for i, key in enumerate(("pert_sens_sel", "pert_pred_sel", "pert_sens_line", "pert_sens_rand")):
        strip(ax, i, col(key, rows), (C_B, "#9aa5b1", C_A, "#52606d")[i])
    ax.set_xticks(range(4), labels)
    ax.set(ylabel="change of the decision variable\nper unit perturbation", title="State perturbations at stimulus onset")

    ax = axes[1, 1]
    amps = np.array([1, 2, 4, 8])
    for name, c, label in (("sel", C_B, "along the selection vector"), ("line", C_A, "along the line attractor"), ("rand", "#52606d", "random directions")):
        errline(ax, amps, [col(f"pert_bias_{name}_a{a}", rows) for a in amps], color=c, label=label)
    ax.set(xscale="log", xlabel="size of the perturbation", ylabel="P(+ | +a) - P(+ | -a)", title="Effect on choices (noisy stimuli)")
    ax.set_xticks(amps, [str(a) for a in amps])
    ax.minorticks_off()
    ax.legend()

    ax = axes[1, 2]
    ax.scatter(col("pert_pred_sel", rows), col("pert_sens_sel", rows), s=9, color=C_B, alpha=0.6, lw=0)
    lim = [0, float(np.nanmax(col("pert_pred_sel", rows))) * 1.05]
    ax.plot(lim, lim, color="k", lw=0.6, ls=":")
    ax.set(xlabel="linear prediction", ylabel="measured", title="Sensitivity along the selection vector")
    panel_labels(axes)
    fig.suptitle("E3: the overlearned feature A is gated more by the selection vector, feature B more by input modulation", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig07_decomposition")


# ----------------------------------------------------------- Fig. 8: switch costs (E4 / H4)
def fig08_switch(data: Data, out: Path) -> None:
    arr, col = data.arr, data.col
    base, over = data.idx("full", 0, 1), data.idx("full", 10, 4)
    cell_rt = arr["cell_rt"][:, TI]                               # [n, transition, task, CSI, RCI]
    cost = cell_rt[:, 1] - cell_rt[:, 0]                          # [n, task, CSI, RCI]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    for ax, axis, xs, xlabel in ((axes[0, 0], 3, CSI_MS, "cue-stimulus interval (ms)"), (axes[0, 1], 2, RCI_MS, "response-cue interval (ms)")):
        by = np.nanmean(cost, axis=axis)                          # [n, task, level]
        for rows, ls, label in ((base, "--", "k = 0, rho = 1"), (over, "-", "k = 10, rho = 4")):
            for task, (c, name) in enumerate(((C_A, "A"), (C_B, "B"))):
                errline(ax, xs, [by[rows][:, task, j] for j in range(len(xs))], color=c, ls=ls, label=f"into task {name}, {label}")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set(xscale="log", xlabel=xlabel, ylabel="switch cost (ms)")
        ax.set_xticks(xs, [str(v) for v in xs])
        ax.minorticks_off()
    axes[0, 0].set_title("Switch cost vs preparation time")
    axes[0, 1].set_title("Switch cost vs time since the last response")
    axes[0, 0].legend(fontsize=6.5)
    grid_lines(axes[0, 2], data, "full", lambda r: (col("switch_cost_A", r), col("switch_cost_B", r)), True)
    axes[0, 2].axhline(0, color="k", lw=0.5, ls=":")
    axes[0, 2].set(ylabel="switch cost (ms)", title="Switch cost into each task")
    axes[0, 2].legend(fontsize=6.5, ncol=2)
    grid_lines(axes[1, 0], data, "full", lambda r: col("switch_asym", r), False)
    axes[1, 0].axhline(0, color="k", lw=0.5, ls=":")
    axes[1, 0].set(ylabel="cost into A - cost into B (ms)", title="Asymmetry of the switch cost")
    axes[1, 0].legend()

    ax = axes[1, 1]
    rows = data.idx("full")
    dist = arr["relax_dist"][rows]                                # [n, context, start, time]
    t = np.arange(dist.shape[-1]) * 10.0
    for i, (label, c) in enumerate((("from the other task's structure (switch)", "#111827"), ("from rest", "#e9a03b"),
                                    ("from the same structure (repeat)", "#2a9d8f"))):
        band(ax, t, np.nanmean(dist[:, :, i], axis=1), c, label)
    for ms in CSI_MS:
        ax.axvline(ms, color="#9aa5b1", lw=0.6, ls=":")
    ax.set(yscale="log", xlabel="time from cue onset (ms)", ylabel="distance to the cued task's slow points", xlim=(0, 1500),
           title="Reconfiguration of the state (no noise)")
    ax.legend(fontsize=6.5)

    ax = axes[1, 2]
    tail = np.concatenate([col("relax_tau_tail_switch_A", rows), col("relax_tau_tail_switch_B", rows)])
    eig = np.concatenate([col("relax_tau_eig2_A", rows), col("relax_tau_eig2_B", rows)])
    ok = np.isfinite(tail) & np.isfinite(eig) & (tail < 1500) & (eig < 1500)
    ax.scatter(eig[ok], tail[ok], s=7, color="#52606d", alpha=0.5, lw=0)
    ax.plot([150, 900], [150, 900], color="k", lw=0.6, ls=":")
    ax.set(xlabel="-tau / Re(lambda) of the slowest non-decision mode (ms)", ylabel="time constant of the approach (ms)",
           title="Relaxation time vs eigenvalue")
    panel_labels(axes)
    fig.suptitle("E4: switch costs are small, vanish with preparation time and are larger into the weaker task B", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig08_switch_costs")


# ------------------------------------------------- Fig. 9: state resets at cue onset (E5 / H5)
def fig09_interventions(data: Data, out: Path) -> None:
    arr, df = data.arr, data.df
    base, over = data.idx("full", 0, 1), data.idx("full", 10, 4)
    e5 = arr["e5_rt"][:, TI]                                       # [n, arm, task, CSI]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    panels = ((axes[0, 0], e5[base].mean(axis=2), "k = 0, rho = 1 (tasks pooled)"), (axes[0, 1], e5[over][:, :, 0], "k = 10, rho = 4: task A"),
              (axes[0, 2], e5[over][:, :, 1], "k = 10, rho = 4: task B"))
    for ax, values, title in panels:                               # values [n, arm, CSI]
        for i, arm in enumerate(ARMS):
            errline(ax, CSI_MS, [values[:, i, j] for j in range(4)], color=C_ARM[arm], label=ARM_LABEL[arm])
        ax.set(xscale="log", xlabel="cue-stimulus interval (ms)", ylabel="response time (ms)", title=title, ylim=(55, 118))
        ax.set_xticks(CSI_MS, [str(v) for v in CSI_MS])
        ax.minorticks_off()
    axes[0, 0].legend(fontsize=6.5)

    ax = axes[1, 0]
    xr = np.log2(RHO_LEVELS)
    cost = np.nanmean(e5[:, 1:] - e5[:, :1], axis=3)               # [n, 3 arms, task]
    for i, arm in enumerate(ARMS[1:]):
        errline(ax, xr + (i - 1) * 0.035, [cost[data.idx("full", rho=rho)][:, i, 0] - cost[data.idx("full", rho=rho)][:, i, 1] for rho in RHO_LEVELS],
                color=C_ARM[arm], label=ARM_LABEL[arm])
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xticks(xr, [str(r) for r in RHO_LEVELS])
    ax.set(xlabel="task ratio rho (A : B trials)", ylabel="cost for A - cost for B (ms)", title="Asymmetry survives every reset\n(all k pooled)")
    ax.legend(fontsize=6.5)

    ax = axes[1, 1]
    comp = {"inertia": e5[:, 1] - e5[:, 3], "reconfiguration": e5[:, 3] - e5[:, 2], "repetition benefit": e5[:, 2] - e5[:, 0]}    # [n, task, CSI]
    groups = [("k0 rho1\nA", base, 0), ("k0 rho1\nB", base, 1), ("k10 rho4\nA", over, 0), ("k10 rho4\nB", over, 1)]
    for i, (name, c) in enumerate(zip(comp, ("#111827", "#e9a03b", "#2a9d8f"))):
        bars(ax, np.arange(4) + (i - 1) * 0.26, [comp[name][rows][:, task, 0] for _, rows, task in groups], c, 0.24, name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(4), [g[0] for g in groups])
    ax.set(ylabel="part of the switch cost (ms)", title="Parts of the switch cost\n(cue 100 ms before the stimulus)")
    ax.legend(fontsize=6.5)

    ax = axes[1, 2]
    long_rt = arr["long_rt"][:, TI]                                # [n, task, RCI, CSI (100, 400)]
    for rows, ls, label in ((base, "--", "k = 0, rho = 1"), (over, "-", "k = 10, rho = 4")):
        for task, (c, name) in enumerate(((C_A, "A"), (C_B, "B"))):
            errline(ax, LONG_RCI_MS, [long_rt[rows][:, task, j, 0] for j in range(4)], color=c, ls=ls, label=f"task {name}, {label}")
    ax.set(xscale="log", xlabel="response-cue interval (ms)", ylabel="response time on repeat trials (ms)",
           title="Repeat trials after a long pause\n(cue 100 ms before the stimulus)")
    ax.set_xticks(LONG_RCI_MS, [str(v) for v in LONG_RCI_MS])
    ax.minorticks_off()
    ax.legend(fontsize=6.5)
    panel_labels(axes)
    fig.suptitle("E5: resetting the state at cue onset shrinks the switch cost but does not remove its asymmetry", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig09_interventions")


# ---------------------------------------------- Fig. 10: Dale's law and SGD (E6 / H6)
def fig10_robustness(data: Data, out: Path) -> None:
    col, df = data.col, data.df
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    measures = (
        (axes[0, 0], lambda r: col("interference_asym", r), np.mean, "interference B - A (ms)", "Asymmetry of the interference"),
        (axes[0, 1], lambda r: col("rt_B", r) - col("rt_A", r), np.mean, "RT B - RT A (ms)", "Response-time difference"),
        (axes[0, 2], lambda r: col("switch_asym", r), np.mean, "switch cost into A - into B (ms)", "Asymmetry of the switch cost"),
        (axes[1, 0], lambda r: col("sel_fraction_A", r) - col("sel_fraction_B", r), np.median, "selection-vector fraction A - B (median)",
         "Decomposition"),
    )
    for ax, getter, stat, ylabel, title in measures:
        grid_lines(ax, data, "full", getter, False, stat=stat, label_prefix="full, ")
        for k in K_LEVELS:                                          # Dale networks: dashed, green shades
            rows = [data.idx("dale", k, rho) for rho in RHO_LEVELS]
            if all(len(r) for r in rows):
                errline(ax, np.log2(RHO_LEVELS) + (K_LEVELS.index(k) - 1) * 0.035 + 0.07, [getter(r) for r in rows], stat,
                        color=("#a7d8d0", "#4fb3a5", "#1f7a6d")[K_LEVELS.index(k)], ls="--", label=f"Dale, k = {k}")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set(ylabel=ylabel, title=title)
    axes[0, 0].legend(fontsize=6.5, ncol=2)

    ax = axes[1, 1]
    groups = [("full\nk0 rho1", "full", 0, 1), ("full\nk10 rho4", "full", 10, 4), ("Dale\nk0 rho1", "dale", 0, 1), ("Dale\nk10 rho4", "dale", 10, 4)]
    x = np.arange(4)
    bars(ax, x - 0.15, [col("interference_asym", data.idx(v, k, rho)) for _, v, k, rho in groups], C_K[10], 0.28, "Adam")
    bars(ax, x + 0.15, [col("interference_asym", data.idx(v, k, rho, optimizer="sgd")) for _, v, k, rho in groups], "#e9a03b", 0.28,
         "SGD with momentum")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set(ylabel="interference B - A (ms)", title="Second optimiser", ylim=(ax.get_ylim()[0], ax.get_ylim()[1] * 1.3))
    ax.legend(loc="upper left", ncol=2)

    ax = axes[1, 2]
    ok = np.ones(len(df), dtype=bool)
    for key in ("slow_mode_ctxA", "slow_mode_ctxB", "others_decay_ctxA", "others_decay_ctxB"):
        ok &= df[key].astype(float).to_numpy() > 0.5
    for variant, ls, colors in (("full", "-", C_K), ("dale", "--", {0: "#a7d8d0", 5: "#4fb3a5", 10: "#1f7a6d"})):
        for k in K_LEVELS:
            share = [100 * ok[data.idx(variant, k, rho)].mean() for rho in RHO_LEVELS]
            ax.plot(np.log2(RHO_LEVELS), share, color=colors[k], ls=ls, marker="o", ms=3.5, label=f"{variant}, k = {k}")
    ax.set_xticks(np.log2(RHO_LEVELS), [str(r) for r in RHO_LEVELS])
    ax.set(xlabel="task ratio rho (A : B trials)", ylabel="networks meeting the criteria (%)", ylim=(0, 105),
           title="Line-attractor criteria in both contexts")
    ax.legend(fontsize=6.5, ncol=2)
    panel_labels(axes)
    fig.suptitle("E6: the same analyses with Dale's law (dashed, green) and with SGD", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig10_robustness")


# -------------------------------------------------- Fig. 11: behaviour during joint training
def fig11_snapshots(data: Data, out: Path) -> None:
    sn = data.snap
    if sn is None:
        print("no snapshots.csv: figure 11 skipped")
        return
    conds = (("full_k0_rho1", "#9aa5b1", "-"), ("full_k0_rho4", "#9aa5b1", "--"), ("full_k10_rho1", "#111827", "-"),
             ("full_k10_rho4", "#111827", "--"), ("full_k5_rho4", "#52606d", "--"))
    sfx = f"@{TH}"
    panels = (("accuracy on task B", lambda g: g["final_acc_B"], False), ("RT B - RT A (ms)", lambda g: g[f"rt_B{sfx}"] - g[f"rt_A{sfx}"], True),
              ("interference B - A (ms)", lambda g: g[f"interference_asym{sfx}"], True),
              ("switch cost into A - into B (ms)", lambda g: g[f"switch_asym{sfx}"], True))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (ylabel, getter, only_included) in zip(axes, panels):
        for name, c, ls in conds:
            g = sn[sn["condition"] == name]
            if only_included:
                g = g[g["included"]]
            epochs = sorted(g["epoch"].unique())
            groups = [getter(g[g["epoch"] == e]).to_numpy(dtype=float) for e in epochs]
            keep = [i for i, v in enumerate(groups) if np.isfinite(v).sum() >= 5]
            if keep:
                errline(ax, [epochs[i] for i in keep], [groups[i] for i in keep], color=c, ls=ls, label=name.replace("full_", "").replace("_", ", "))
        ax.set(xlabel="epochs of joint training", ylabel=ylabel)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 2, 5, 10, 20], ["0", "1", "2", "5", "10", "20"])
        if only_included:
            ax.axhline(0, color="k", lw=0.5, ls=":")
    axes[0].legend(fontsize=6.5)
    axes[0].set_title("Learning task B")
    axes[1].set_title("Speed advantage of task A")
    axes[2].set_title("Asymmetry of the interference")
    axes[3].set_title("Asymmetry of the switch cost")
    panel_labels(axes)
    fig.tight_layout()
    save(fig, out, "fig11_training_time_course")


FIGURES = {1: fig01_validation, 2: fig02_task_training, 3: fig03_replication, 4: fig04_state_space, 5: fig05_interference,
           6: fig06_mechanism, 7: fig07_decomposition, 8: fig08_switch, 9: fig09_interventions, 10: fig10_robustness,
           11: fig11_snapshots}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="figure numbers (default: all)")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    data = Data(ROOT / args.results)
    failed = []
    for number, func in FIGURES.items():
        if args.only and number not in args.only:
            continue
        try:
            func(data, out)
        except Exception as exc:                                  # keep going: one broken panel must not block the rest
            import traceback

            traceback.print_exc()
            failed.append(f"fig {number}: {exc}")
    if failed:
        raise SystemExit("failed: " + "; ".join(failed))


if __name__ == "__main__":
    main()
