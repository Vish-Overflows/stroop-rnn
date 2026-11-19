"""Experiments E1-E5 on trained networks.

Every function takes a BatchedRNN holding the networks of one training condition and returns a
dictionary of NumPy arrays whose first dimension runs over networks. All test inputs come from fixed
seeds that do not depend on the training condition: network i of every condition is tested on the
same trials, while the networks of one condition get independent trials (so the sampling error of
the test set is part of the variability across networks).

    behaviour_battery     E2, E4  interference, facilitation and switch costs on back-to-back sequences
    psychometric          E1      context-dependent choice as a function of both feature values
    perturbations         E3      state perturbations along the selection vector vs other directions
    relaxation            E4      time for the state to move between the contexts' slow-point structures
    interventions         E5      state resets at cue onset on switch trials
    long_rsi_repeats      E5      repeat trials after long response-stimulus intervals
"""

from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import torch
from scipy.special import expit

from .behavior import cell_means, responses, stroop_measures, switch_measures, task_accuracy
from .config import RT_THRESHOLDS, TaskConfig
from .dynamics import CUES, OP_CSI_MS, flow
from .fixedpoints import RNNField
from .model import BatchedRNN, NoiseSource
from .stats import fit_logistic
from .task import TASK_A, TASK_B, Sequences, make_sequences

TEST_CSI_MS = (100.0, 200.0, 400.0, 800.0)
TEST_RCI_MS = (100.0, 200.0, 400.0, 600.0)
TEST_SEED = 771_000            # all test inputs derive from this seed


# ------------------------------------------------------------------------------- helpers
def _labels(seq: Sequences) -> Sequences:
    """The trial labels of a batch without the time series (to keep many batches in memory)."""
    return replace(seq, x=torch.empty(0), target=torch.empty(0))


def _concat(parts: list[Sequences]) -> Sequences:
    out = {}
    for f in fields(Sequences):
        vals = [getattr(p, f.name) for p in parts]
        out[f.name] = vals[0] if f.name in ("x", "target") else torch.cat(vals, dim=1)
    return Sequences(**out)


def _concat_resp(parts: list[dict]) -> dict:
    return {
        "rt": {th: torch.cat([p["rt"][th] for p in parts], dim=1) for th in parts[0]["rt"]},
        "resp_correct": {th: torch.cat([p["resp_correct"][th] for p in parts], dim=1) for th in parts[0]["resp_correct"]},
        "premature": {th: torch.cat([p["premature"][th] for p in parts], dim=1) for th in parts[0]["premature"]},
        "final_correct": torch.cat([p["final_correct"] for p in parts], dim=1),
    }


def _stack(per_network: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(per_network)).float()


# ------------------------------------------------------------------- E2 / E4: behaviour
def battery_versions(cfg: TaskConfig, shape: tuple[int, int], gen: torch.Generator):
    """Nine versions of a batch of random test sequences that share timing, relevant features and noise.

    Random sequences are drawn as in training, with both tasks equally likely on every trial. Each of
    them is then presented in 3 x 3 versions:

      * the irrelevant feature of every trial is rotated through its three values, so every trial
        occurs once as congruent, once as neutral and once as incongruent;
      * the cued task is as drawn, or flipped on the 1st/3rd/5th trials, or flipped on the 2nd/4th/6th
        trials. Flipping every other trial turns every switch into a repeat and vice versa, so every
        trial occurs after a switch and after a repeat (and the context sequence is still a sequence of
        independent fair coin flips).

    Each condition of a comparison therefore contains the same trials with the same stimulus noise,
    which removes most of the trial-to-trial variability from the differences of response times.
    """
    base = make_sequences(cfg, shape, gen, csi_choices=TEST_CSI_MS, rci_choices=TEST_RCI_MS, sigma_in=0.0)
    T = base.n_steps
    noise = torch.randn(T, *shape, 2, generator=gen)
    position = torch.arange(base.context.shape[-1])
    level = torch.round(base.irr / cfg.signal).long() + 1                 # 0, 1, 2
    for flip in range(3):
        flipped = (position % 2 == flip - 1) if flip else torch.zeros_like(position, dtype=torch.bool)
        context = torch.where(flipped, 1 - base.context, base.context)
        for rot in range(3):
            irr = ((level + rot) % 3 - 1).float() * cfg.signal
            yield make_sequences(cfg, shape, gen, context=context, rel=base.rel, irr=irr, csi=base.csi,
                                 rci=base.rci, hold=base.hold, horizon=T, noise=noise)


def behaviour_battery(model: BatchedRNN, cfg: TaskConfig, n_base: int = 400, chunk: int = 100,
                      seed: int = TEST_SEED, thresholds=RT_THRESHOLDS) -> dict:
    """Stroop and task-switching measures on back-to-back sequences (see battery_versions).

    The cue-stimulus and response-cue intervals take four levels each; the first trial of every
    sequence is a warm-up that is not analysed. Returns per-network arrays:

        final_acc_A, final_acc_B           final-choice accuracy (used for the 90% criterion)
        "<measure>@<theta>"                every entry of stroop_measures and switch_measures
        "<measure>@<theta>:raw"            the same without the exclusion of premature responses
        cell_rt, cell_acc, cell_premature, cell_n
                                           [S, theta, transition, task, CSI, RCI] cell means for the mixed model
        cong_rt, cong_acc                  [S, theta, transition, task, congruency]
        stroop_rt, stroop_acc              [S, theta, task, congruency, CSI]
    """
    S = model.n_networks
    gen = torch.Generator().manual_seed(seed)
    resp_parts, label_parts = [], []
    for i in range(0, n_base, chunk):
        b = min(chunk, n_base - i)
        for seq in battery_versions(cfg, (S, b), gen):
            y = model.simulate(seq.x, sigma_rec=cfg.sigma_rec, noise=NoiseSource(seed + 1 + i))
            resp_parts.append(responses(y, seq, cfg, thresholds))
            label_parts.append(_labels(seq))
    resp, seq = _concat_resp(resp_parts), _concat(label_parts)
    analysed = seq.valid & (seq.switch >= 0)

    acc_a, acc_b = task_accuracy(resp, seq)
    out = {"final_acc_A": acc_a, "final_acc_B": acc_b, "n_trials": analysed.float().sum(dim=(1, 2)).numpy()}
    csi_levels = [cfg.steps(ms) for ms in TEST_CSI_MS]
    rci_levels = [cfg.steps(ms) for ms in TEST_RCI_MS]
    n_th = len(thresholds)
    cell_rt = np.full((S, n_th, 2, 2, len(csi_levels), len(rci_levels)), np.nan)
    cell_acc, cell_premature, cell_n = np.full_like(cell_rt, np.nan), np.full_like(cell_rt, np.nan), np.zeros_like(cell_rt)
    cong_rt = np.full((S, n_th, 2, 2, 3), np.nan)
    cong_acc = np.full_like(cong_rt, np.nan)
    stroop_rt = np.full((S, n_th, 2, 3, len(csi_levels)), np.nan)
    stroop_acc = np.full_like(stroop_rt, np.nan)
    for ti, theta in enumerate(thresholds):
        for exclude, suffix in ((True, ""), (False, ":raw")):
            measures = {**stroop_measures(resp, seq, theta, analysed, exclude), **switch_measures(resp, seq, theta, analysed, exclude)}
            for name, val in measures.items():
                out[f"{name}@{theta}{suffix}"] = np.asarray(val)
        nan_rt = torch.isnan(resp["rt"][theta]) & analysed
        out[f"no_response@{theta}"] = (nan_rt.float().sum(dim=(1, 2)) / analysed.float().sum(dim=(1, 2))).numpy()
        for sw in (0, 1):
            for task in (TASK_A, TASK_B):
                base = analysed & (seq.switch == sw) & (seq.context == task)
                for ci, c in enumerate(csi_levels):
                    for ri, r in enumerate(rci_levels):
                        cell = cell_means(resp, seq, base & (seq.csi == c) & (seq.rci == r), theta)
                        cell_rt[:, ti, sw, task, ci, ri] = cell["rt"].numpy()
                        cell_acc[:, ti, sw, task, ci, ri] = cell["resp_acc"].numpy()
                        cell_premature[:, ti, sw, task, ci, ri] = cell["premature"].numpy()
                        cell_n[:, ti, sw, task, ci, ri] = cell["n"].numpy()
                for cong in range(3):
                    cell = cell_means(resp, seq, base & (seq.congruency == cong), theta)
                    cong_rt[:, ti, sw, task, cong] = cell["rt"].numpy()
                    cong_acc[:, ti, sw, task, cong] = cell["resp_acc"].numpy()
        for task in (TASK_A, TASK_B):
            for cong in range(3):
                for ci, c in enumerate(csi_levels):
                    cell = cell_means(resp, seq, analysed & (seq.context == task) & (seq.congruency == cong) & (seq.csi == c), theta)
                    stroop_rt[:, ti, task, cong, ci] = cell["rt"].numpy()
                    stroop_acc[:, ti, task, cong, ci] = cell["resp_acc"].numpy()
    out.update(cell_rt=cell_rt, cell_acc=cell_acc, cell_premature=cell_premature, cell_n=cell_n,
               cong_rt=cong_rt, cong_acc=cong_acc, stroop_rt=stroop_rt, stroop_acc=stroop_acc)
    return out


# -------------------------------------------------------------------- E1: psychometric curves
PSY_LEVELS = (-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0)     # feature values in units of s


def psychometric(model: BatchedRNN, cfg: TaskConfig, h_neutral: np.ndarray, n_rep: int = 40, chunk_rep: int = 10,
                 seed: int = TEST_SEED + 10_000, csi_ms: float = 400.0) -> dict:
    """Choices in single trials for all combinations of graded values of both features, in both contexts.

    Every trial starts from the network's resting state. Returns per network

        p_plus    [S, context, value of A, value of B]  probability of choosing output 1 ("+")
        beta, se  [S, context, 3]  logistic regression of the choice on (1, u_A, u_B)
        n_per_cell
    """
    S, N = model.n_networks, model.n_units
    levels = torch.tensor(PSY_LEVELS) * cfg.signal
    L = len(levels)
    ctx_g, a_g, b_g = torch.meshgrid(torch.arange(2.0), levels, levels, indexing="ij")
    ctx_g, a_g, b_g = ctx_g.reshape(-1).long(), a_g.reshape(-1), b_g.reshape(-1)       # 2 L^2 conditions
    gen = torch.Generator().manual_seed(seed)
    h0_all = torch.from_numpy(h_neutral).float()
    count = torch.zeros(S, len(ctx_g))
    for i in range(0, n_rep, chunk_rep):
        reps = min(chunk_rep, n_rep - i)
        B = reps * len(ctx_g)
        context = ctx_g.repeat(reps).view(1, B, 1).expand(S, -1, -1)
        val_a = a_g.repeat(reps).view(1, B, 1).expand(S, -1, -1)
        val_b = b_g.repeat(reps).view(1, B, 1).expand(S, -1, -1)
        rel = torch.where(context == TASK_A, val_a, val_b)
        irr = torch.where(context == TASK_A, val_b, val_a)
        seq = make_sequences(
            cfg, (S, B), gen, n_trials=1, context=context, rel=rel, irr=irr,
            csi=torch.full((S, B, 1), cfg.steps(csi_ms)), rci=torch.full((S, B, 1), 1),
            hold=torch.full((S, B, 1), cfg.steps(cfg.hold_ms[0])), tail_steps=0)
        y = model.simulate(seq.x, sigma_rec=cfg.sigma_rec, noise=NoiseSource(seed + 1 + i),
                           h0=h0_all.view(S, 1, N).expand(-1, B, -1))
        d_last = (y[..., 0] - y[..., 1]).permute(1, 2, 0).gather(2, seq.cue_off - 1)[..., 0]     # [S, B]
        count += (d_last > 0).float().view(S, reps, -1).sum(1)
    p_plus = (count / n_rep).view(S, 2, L, L).numpy()

    beta = np.empty((S, 2, 3))
    se = np.empty((S, 2, 3))
    x_cell = np.column_stack([a_g.numpy(), b_g.numpy()]).reshape(2, L * L, 2)
    for s in range(S):
        for c in range(2):
            # one row per trial: n_rep trials per cell, "+" choices first
            k = np.rint(p_plus[s, c].reshape(-1) * n_rep).astype(int)
            x = np.repeat(x_cell[c], n_rep, axis=0)
            y01 = np.concatenate([np.r_[np.ones(ki), np.zeros(n_rep - ki)] for ki in k])
            beta[s, c], se[s, c] = fit_logistic(x, y01, ridge=1e-6)
    return {"p_plus": p_plus, "beta": beta, "se": se, "n_per_cell": n_rep, "levels": np.asarray(PSY_LEVELS)}


# -------------------------------------------------------------------- E3: state perturbations
def perturbations(model: BatchedRNN, cfg: TaskConfig, details: list[dict], amplitudes=(1.0, 2.0, 4.0, 8.0),
                  n_random: int = 8, n_noise: int = 48, seed: int = TEST_SEED + 20_000) -> dict:
    """Perturb the state at stimulus onset along the selection vector, the line attractor or random directions.

    All perturbations are unit vectors in the space of the state h, multiplied by +-amplitude. A state
    change dh changes the rates by G dh, so its projection on the decision mode is l . G dh: the
    direction with the largest effect per unit norm is G l (direction 0). Direction 1 displaces the
    rates along the line attractor itself (G^-1 r); the remaining directions are random and oriented
    so that they have a positive overlap with G l.

    Two measurements, both from the prepared state (cue on for the longest cue-stimulus interval):
      * no stimulus, no noise: change of the decision variable at the end of the stimulus period;
      * zero-mean noisy stimulus (the same noise in every arm): probability of the "+" choice.

    Returns per network [S, context, direction, amplitude, sign] arrays dd_free and p_plus, the
    unperturbed p_plus0 [S, context] and the linear prediction dd_pred [S, context, direction] per unit amplitude.
    """
    S, N = model.n_networks, model.n_units
    rng = np.random.default_rng(seed)
    amps = np.asarray(amplitudes)
    D, A = 2 + n_random, len(amps)
    stim, hold = cfg.steps(cfg.stim_ms), cfg.steps(cfg.hold_ms[0])
    T = stim + hold
    dd_free = np.empty((S, 2, D, A, 2))
    p_plus = np.empty((S, 2, D, A, 2))
    p_plus0 = np.empty((S, 2))
    dd_pred = np.empty((S, 2, D))
    overlap = np.empty((S, 2, D))
    gen = torch.Generator().manual_seed(seed + 1)
    stim_noise = cfg.sigma_in * torch.randn(stim, S, n_noise, 2, generator=gen)
    for c, cue in enumerate(("A", "B")):
        # directions and start states
        dirs = np.empty((S, D, N))
        h_prep = np.stack([det["ctx"][cue]["h_op"][-1] for det in details])
        for s, det in enumerate(details):
            ctx = det["ctx"][cue]
            gains = expit(h_prep[s])
            sel = gains * ctx["l"]
            line = ctx["r"] / gains
            rand = rng.standard_normal((n_random, N))
            rand *= np.sign(rand @ sel)[:, None]
            dirs[s] = np.vstack([sel, line, rand])
            dirs[s] /= np.linalg.norm(dirs[s], axis=1, keepdims=True)
            proj = (dirs[s] * gains) @ ctx["l"]                       # l . G dh per unit amplitude
            dd_pred[s, c] = ctx["wr"] * proj * math.exp(ctx["lam_mean"] * stim * cfg.dt / cfg.tau)
            overlap[s, c] = dirs[s] @ dirs[s][0]
        signed = np.concatenate([amps, -amps])                         # [2A]
        offsets = dirs[:, :, None, :] * signed[None, None, :, None]    # [S, D, 2A, N]
        starts = h_prep[:, None, None, :] + offsets
        starts = np.concatenate([h_prep[:, None, :], starts.reshape(S, D * 2 * A, N)], axis=1)    # arm 0 = unperturbed
        n_arms = starts.shape[1]
        h0 = torch.from_numpy(starts).float()

        # (1) noise-free, no stimulus
        x = torch.zeros(T, S, n_arms, 4)
        x[..., 2 + c] = 1.0
        y = model.simulate(x, h0=h0)
        d_end = (y[stim - 1, :, :, 0] - y[stim - 1, :, :, 1]).numpy()                 # [S, arms]
        dd = (d_end[:, 1:] - d_end[:, :1]).reshape(S, D, 2, A)
        dd_free[:, c] = dd.transpose(0, 1, 3, 2)

        # (2) zero-mean noisy stimulus, identical across arms
        x = torch.zeros(T, S, n_arms, n_noise, 4)
        x[..., 2 + c] = 1.0
        x[:stim, :, :, :, :2] = stim_noise.unsqueeze(2)
        x = x.reshape(T, S, n_arms * n_noise, 4)
        h0n = h0.unsqueeze(2).expand(-1, -1, n_noise, -1).reshape(S, n_arms * n_noise, N)
        y = model.simulate(x, sigma_rec=cfg.sigma_rec, noise=NoiseSource(seed + 2 + c), h0=h0n)
        plus = ((y[-1, :, :, 0] - y[-1, :, :, 1]) > 0).float().view(S, n_arms, n_noise).mean(-1).numpy()
        p_plus0[:, c] = plus[:, 0]
        p_plus[:, c] = plus[:, 1:].reshape(S, D, 2, A).transpose(0, 1, 3, 2)
    return {"dd_free": dd_free, "p_plus": p_plus, "p_plus0": p_plus0, "dd_pred": dd_pred, "overlap": overlap,
            "amplitudes": amps}


# ------------------------------------------- E2 / E3: noise-free response to the two features
STEP_CSI_MS = (100.0, 800.0)
STEP_VALUES = (-1.0, 0.0, 1.0)


def step_responses(model: BatchedRNN, cfg: TaskConfig, details: list[dict], csi_ms=STEP_CSI_MS) -> dict:
    """Noise-free response of the decision variable to constant stimuli, from the prepared state.

    Every network starts from its resting state, sees the cue for one of the cue-stimulus intervals and
    then a constant stimulus (u_A, u_B) from the 3 x 3 grid of feature values; there is neither stimulus
    noise nor private noise. This isolates how strongly each feature drives the decision variable in
    each context, at the short times that determine the response time and at the end of the stimulus.

    Returns per network
        d       [S, context, CSI, u_A, u_B, W]   y1 - y2 at the W steps of the response window
        d_pre   [S, context, CSI, u_A, u_B]      y1 - y2 at the last step before the stimulus
        slope   [S, context, feature]            linear prediction of the initial slope of the decision variable
                                                 (per ms, for a unit step of the feature): w_d . G w_in,k / tau
                                                 with the gains G = phi'(h) of the state prepared for the longest CSI
    """
    S, N = model.n_networks, model.n_units
    W = cfg.window_steps
    stim = cfg.steps(cfg.stim_ms)
    values = torch.tensor(STEP_VALUES) * cfg.signal
    ctx_g, a_g, b_g = torch.meshgrid(torch.arange(2.0), values, values, indexing="ij")
    ctx_g, a_g, b_g = ctx_g.reshape(-1).long(), a_g.reshape(-1), b_g.reshape(-1)            # 18 conditions
    B = len(ctx_g)
    h0 = _stack([det["h_neutral"] for det in details]).view(S, 1, N).expand(-1, B, -1)
    d = np.empty((S, 2, len(csi_ms), 3, 3, W))
    d_pre = np.empty((S, 2, len(csi_ms), 3, 3))
    prepared = None
    for ci, ms in enumerate(csi_ms):
        csi = cfg.steps(ms)
        x = torch.zeros(csi + W, S, B, 4)
        x[:, :, torch.arange(B), 2 + ctx_g] = 1.0
        x[csi:csi + stim, :, :, 0] = a_g
        x[csi:csi + stim, :, :, 1] = b_g
        y, hs = model.simulate(x, h0=h0, return_h=True)
        dv = (y[..., 0] - y[..., 1]).permute(1, 2, 0).numpy()                             # [S, B, T]
        d[:, :, ci] = dv[:, :, csi:].reshape(S, 2, 3, 3, W)
        d_pre[:, :, ci] = dv[:, :, csi - 1].reshape(S, 2, 3, 3)
        if ms == max(csi_ms):
            prepared = hs[csi - 1].view(S, 2, 9, N)[:, :, 0]                              # [S, context, N]
    gains = torch.sigmoid(prepared.double())
    w_d = (model.w_out[:, 0] - model.w_out[:, 1]).detach().double()                         # [S, N]
    w_in = model.w_in.detach().double()                                                     # [S, N, 2]
    slope = torch.einsum("sn,scn,snk->sck", w_d, gains, w_in).numpy() / cfg.tau

    # the same stimuli without any cue, from the resting state (which task does the network perform by default?)
    x = torch.zeros(W, S, 9, 4)
    x[:stim, :, :, 0] = a_g[:9]
    x[:stim, :, :, 1] = b_g[:9]
    y = model.simulate(x, h0=h0[:, :9])
    d_nocue = (y[..., 0] - y[..., 1]).permute(1, 2, 0).numpy().reshape(S, 3, 3, W)
    rest = torch.nn.functional.softplus(h0[:, 0].double())
    d_rest = torch.einsum("sn,sn->s", w_d, rest).numpy()
    return {"d": d, "d_pre": d_pre, "slope": slope, "d_nocue": d_nocue, "d_rest": d_rest,
            "csi_ms": np.asarray(csi_ms), "values": np.asarray(STEP_VALUES)}


def crossing_time(d: np.ndarray, d_pre: np.ndarray, theta: float, dt: float) -> np.ndarray:
    """First time the traces d [..., W] (samples at dt, 2 dt, ...; value d_pre at time 0) exceed theta.

    The crossing is located by linear interpolation between samples; NaN if the threshold is not reached.
    """
    above = d > theta
    reached = above.any(-1)
    first = above.argmax(-1)
    d1 = np.take_along_axis(d, first[..., None], -1)[..., 0]
    d0 = np.where(first > 0, np.take_along_axis(d, np.maximum(first - 1, 0)[..., None], -1)[..., 0], d_pre)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.clip((theta - d0) / (d1 - d0), 0.0, 1.0)
    return np.where(reached, (first + frac) * dt, np.nan)


def step_summary(step: dict, cfg: TaskConfig, theta: float) -> dict:
    """Scalars derived from step_responses, per network and task (index 0 = task A, 1 = task B).

        resp_rel, resp_irr  [S, task, CSI, W]  antisymmetric response of d to the relevant / irrelevant feature alone
        rt                  [S, task, CSI, 3]  noise-free response time for congruent, neutral, incongruent stimuli
        interference, facilitation  [S, task, CSI]  differences of these response times
        slope_rel, slope_irr [S, task]         predicted initial slopes (per ms)
        default             [S, feature, W]    antisymmetric response of d to each feature alone when no task is cued
    """
    d, d_pre = step["d"], step["d_pre"]                    # [S, context, CSI, u_A, u_B, W]
    S = d.shape[0]
    n_csi, W = d.shape[2], d.shape[-1]
    resp_rel = np.empty((S, 2, n_csi, W))
    resp_irr = np.empty((S, 2, n_csi, W))
    rt = np.empty((S, 2, n_csi, 3))
    for task in (TASK_A, TASK_B):
        # move the relevant feature to the first stimulus axis
        dt_, pre = (d[:, task], d_pre[:, task]) if task == TASK_A else (d[:, task].swapaxes(2, 3), d_pre[:, task].swapaxes(2, 3))
        resp_rel[:, task] = 0.5 * (dt_[:, :, 2, 1] - dt_[:, :, 0, 1])
        resp_irr[:, task] = 0.5 * (dt_[:, :, 1, 2] - dt_[:, :, 1, 0])
        for cong, sign in enumerate((1, 0, -1)):            # congruent, neutral, incongruent
            plus = crossing_time(dt_[:, :, 2, 1 + sign], pre[:, :, 2, 1 + sign], theta, cfg.dt)
            minus = crossing_time(-dt_[:, :, 0, 1 - sign], -pre[:, :, 0, 1 - sign], theta, cfg.dt)
            rt[:, task, :, cong] = 0.5 * (plus + minus)
    slope = step["slope"]                                  # [S, context, feature]
    out = {
        "resp_rel": resp_rel, "resp_irr": resp_irr, "rt": rt,
        "interference": rt[..., 2] - rt[..., 1], "facilitation": rt[..., 1] - rt[..., 0],
        "slope_rel": np.stack([slope[:, 0, 0], slope[:, 1, 1]], axis=1),
        "slope_irr": np.stack([slope[:, 0, 1], slope[:, 1, 0]], axis=1),
    }
    if "d_nocue" in step:
        nocue = step["d_nocue"]                            # [S, u_A, u_B, W]
        out["default"] = np.stack([0.5 * (nocue[:, 2, 1] - nocue[:, 0, 1]), 0.5 * (nocue[:, 1, 2] - nocue[:, 1, 0])], axis=1)
    return out


# ------------------------------------------------------- E4: moving between the two structures
def _distance_to_segment(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance of every state in h [T, N] to the straight segment fitted to the slow points pts [P, N]."""
    centre = pts.mean(0)
    u = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
    t_pts = (pts - centre) @ u
    t = np.clip((h - centre) @ u, t_pts.min(), t_pts.max())
    return np.linalg.norm(h - centre - t[:, None] * u, axis=1)


def _first_below(values: np.ndarray, level: float, dt: float) -> float:
    idx = np.flatnonzero(values <= level)
    return float(idx[0] * dt) if idx.size else float("nan")


def relaxation(model: BatchedRNN, cfg: TaskConfig, details: list[dict], horizon_ms: float = 2000.0) -> dict:
    """Noise-free time course of the state after cue onset, relative to the cued context's slow points.

    For both target contexts and three starting states,
        "switch"  the centre of the other context's slow points (previous task set fully active),
        "rest"    the resting state without cue,
        "repeat"  the centre of the target context's own slow points after 100 ms without cue,
    the function follows the state with the cue on and measures its distance to the target context's
    line of slow points. Returns per network [S, context, start] arrays:

        dist0, t_half, t_e         initial distance, time to fall to 1/2 and to 1/e of it (ms)
        tau_tail                   time constant of the exponential decay of the distance from 300 to 800 ms
        dist_csi [..., 4]          distance at the four cue-stimulus intervals
    and [S, context] arrays tau_eig2 = -tau / Re(lambda_2), the time constant of the slowest mode other
    than the decision mode at the target structure, and tau_eig_rest, the same at the resting state.
    """
    S = model.n_networks
    w_rec = model.w_rec().detach().double().numpy()
    w_ctx = model.w_ctx.detach().double().numpy()
    steps = cfg.steps(horizon_ms)
    csi_steps = [cfg.steps(ms) for ms in OP_CSI_MS]
    starts = ("switch", "rest", "repeat")
    out = {k: np.full((S, 2, len(starts)), np.nan) for k in ("dist0", "t_half", "t_e", "tau_tail")}
    out["dist_csi"] = np.full((S, 2, len(starts), len(csi_steps)), np.nan)
    out["dist"] = np.full((S, 2, len(starts), steps + 1), np.nan, dtype=np.float32)      # the whole time course
    out["tau_eig2"] = np.full((S, 2), np.nan)
    out["tau_decision"] = np.full((S, 2), np.nan)
    lo, hi = cfg.steps(300.0), cfg.steps(800.0)
    for s, det in enumerate(details):
        off = RNNField(w_rec[s], np.zeros(w_rec.shape[1]), cfg.tau)
        for c, cue in enumerate(("A", "B")):
            other = "B" if cue == "A" else "A"
            ctx = det["ctx"][cue]
            field = RNNField(w_rec[s], w_ctx[s] @ np.asarray(CUES[cue]), cfg.tau)
            h0 = {
                "switch": det["ctx"][other]["h_line"],
                "rest": det["h_neutral"],
                "repeat": flow(off, ctx["h_line"], cfg.steps(100.0), cfg.dt)[-1],
            }
            for i, name in enumerate(starts):
                dist = _distance_to_segment(flow(field, h0[name], steps, cfg.dt), ctx["valley"])
                out["dist"][s, c, i, :len(dist)] = dist[:steps + 1]
                out["dist0"][s, c, i] = dist[0]
                out["t_half"][s, c, i] = _first_below(dist, 0.5 * dist[0], cfg.dt)
                out["t_e"][s, c, i] = _first_below(dist, dist[0] / math.e, cfg.dt)
                out["dist_csi"][s, c, i] = dist[csi_steps]
                seg = dist[lo:hi + 1]
                if np.all(seg > 1e-9):
                    slope = np.polyfit(np.arange(lo, hi + 1) * cfg.dt, np.log(seg), 1)[0]
                    out["tau_tail"][s, c, i] = -1.0 / slope if slope < 0 else float("inf")
            out["tau_eig2"][s, c] = -cfg.tau / ctx["lam_second"] if ctx["lam_second"] < 0 else float("inf")
            out["tau_decision"][s, c] = -cfg.tau / ctx["lam"] if ctx["lam"] < 0 else float("inf")
    out["starts"] = np.asarray(starts)
    return out


# ------------------------------------------------------------- E5: state resets at cue onset
E5_ARMS = ("repeat", "switch", "switch_to_target", "switch_to_neutral")
E5_RCI_MS = (100.0, 600.0, 2000.0, 5000.0)
E5_LONG_CSI_MS = (100.0, 400.0)


def _probe_grid(cfg: TaskConfig, levels: dict, n_rep: int):
    """Full factorial of probe-trial variables, repeated n_rep times. Returns a dict of 1-D tensors."""
    names = list(levels)
    grids = torch.meshgrid(*[torch.as_tensor(levels[k], dtype=torch.float32) for k in names], indexing="ij")
    out = {k: g.reshape(-1).repeat(n_rep) for k, g in zip(names, grids)}
    out["context"], out["csi"] = out["context"].long(), out["csi"].long()
    return out


def _pair_sequences(cfg: TaskConfig, S: int, probe: dict, prime_context: torch.Tensor, prime_ab: torch.Tensor,
                    rci2: torch.Tensor, seed: int) -> Sequences:
    """Prime -> probe pairs for every network.

    probe: 1-D tensors [B] (the same factorial design for every network); prime_context [B];
    prime_ab [S, B, 2]: the prime stimulus (u_A, u_B), which is the same whatever the prime's context.
    """
    B = len(probe["context"])

    def both(prime: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.stack([prime.expand(S, B), second.expand(S, B)], dim=2)

    prime_is_a = (prime_context == TASK_A).expand(S, B)
    context = both(prime_context, probe["context"])
    rel = both(torch.where(prime_is_a, prime_ab[..., 0], prime_ab[..., 1]), probe["rel"])
    irr = both(torch.where(prime_is_a, prime_ab[..., 1], prime_ab[..., 0]), probe["irr"])
    csi = both(torch.full((B,), cfg.steps(400.0)), probe["csi"])
    rci = both(torch.full((B,), cfg.steps(200.0)), rci2)
    hold = both(torch.full((B,), cfg.steps(500.0)), torch.full((B,), cfg.steps(cfg.hold_ms[0])))
    gen = torch.Generator().manual_seed(seed)                 # only the stimulus noise is drawn from it
    horizon = int((rci + csi + hold).sum(-1).max()) + 2 * cfg.steps(cfg.stim_ms) + 1
    return make_sequences(cfg, (S, B), gen, n_trials=2, context=context, rel=rel, irr=irr, csi=csi, rci=rci,
                          hold=hold, horizon=horizon)


def _probe_cells(resp: dict, seq: Sequences, cells: list[torch.Tensor], thresholds, shape: tuple) -> dict:
    """Probe-trial means in each cell, reshaped to [S, theta, *shape]: RT of correct responses, response
    accuracy and rate of premature responses, plus the RT without the exclusion of premature responses."""
    S = seq.context.shape[0]
    out = {k: np.full((S, len(thresholds), len(cells)), np.nan) for k in ("rt", "acc", "premature", "rt_raw")}
    probe = torch.zeros_like(seq.valid)
    probe[..., 1] = True
    for ti, theta in enumerate(thresholds):
        for ci, cell in enumerate(cells):
            m = cell_means(resp, seq, probe & cell.view(1, -1, 1), theta)
            out["rt"][:, ti, ci], out["acc"][:, ti, ci] = m["rt"].numpy(), m["resp_acc"].numpy()
            out["premature"][:, ti, ci] = m["premature"].numpy()
            out["rt_raw"][:, ti, ci] = cell_means(resp, seq, probe & cell.view(1, -1, 1), theta, False)["rt"].numpy()
    return {k: v.reshape(S, len(thresholds), *shape) for k, v in out.items()}


def interventions(model: BatchedRNN, cfg: TaskConfig, details: list[dict], n_rep: int = 60, chunk_rep: int = 20,
                  seed: int = TEST_SEED + 30_000, thresholds=RT_THRESHOLDS) -> dict:
    """E5 (a): at the cue onset of a switch trial the state is left alone, set to the centre of the target
    context's slow points, or set to the resting state; repeat trials without intervention are the baseline.

    Prime and probe follow each other after the shortest response-cue interval (100 ms). The four arms
    use the same probe trials, the same prime stimuli and the same noise; only the prime's context and
    the intervention differ. Returns rt, acc, premature and rt_raw as [S, theta, arm, task, CSI].
    """
    S, N = model.n_networks, model.n_units
    csi_levels = [cfg.steps(ms) for ms in TEST_CSI_MS]
    levels = {"context": [TASK_A, TASK_B], "csi": csi_levels, "rel": [-cfg.signal, cfg.signal],
              "irr": [-cfg.signal, 0.0, cfg.signal]}
    h_neutral = _stack([det["h_neutral"] for det in details])                              # [S, N]
    h_line = torch.stack([_stack([det["ctx"][cue]["h_line"] for det in details]) for cue in ("A", "B")], dim=1)
    rng = torch.Generator().manual_seed(seed)
    sums = {arm: [] for arm in E5_ARMS}
    for i in range(0, n_rep, chunk_rep):
        probe = _probe_grid(cfg, levels, min(chunk_rep, n_rep - i))
        B = len(probe["context"])
        prime_ab = cfg.signal * (2.0 * torch.randint(0, 2, (S, B, 2), generator=rng) - 1.0)
        rci2 = torch.full((B,), cfg.steps(100.0))
        target_state = h_line[:, probe["context"]]                                        # [S, B, N]
        neutral_state = h_neutral.view(S, 1, N).expand(-1, B, -1)
        everything = torch.ones(S, B, dtype=torch.bool)
        for arm in E5_ARMS:
            prime_context = probe["context"] if arm == "repeat" else 1 - probe["context"]
            seq = _pair_sequences(cfg, S, probe, prime_context, prime_ab, rci2, seed + 1 + i)
            t_cue = int(seq.cue_on[0, 0, 1])
            assert bool((seq.cue_on[..., 1] == t_cue).all())
            iv = None
            if arm == "switch_to_target":
                iv = {t_cue: (everything, target_state)}
            elif arm == "switch_to_neutral":
                iv = {t_cue: (everything, neutral_state)}
            y = model.simulate(seq.x, sigma_rec=cfg.sigma_rec, noise=NoiseSource(seed + 2 + i), interventions=iv)
            sums[arm].append((responses(y, seq, cfg, thresholds), _labels(seq)))
    per_arm = []
    for arm in E5_ARMS:
        resp = _concat_resp([r for r, _ in sums[arm]])
        seq = _concat([lab for _, lab in sums[arm]])
        ctx, csi = seq.context[0, :, 1], seq.csi[0, :, 1]
        cells = [(ctx == task) & (csi == c) for task in (TASK_A, TASK_B) for c in csi_levels]
        per_arm.append(_probe_cells(resp, seq, cells, thresholds, (2, len(csi_levels))))
    out = {k: np.stack([m[k] for m in per_arm], axis=2) for k in per_arm[0]}       # [S, theta, arm, task, CSI]
    out.update(arms=np.asarray(E5_ARMS), csi_ms=np.asarray(TEST_CSI_MS))
    return out


def long_rsi_repeats(model: BatchedRNN, cfg: TaskConfig, n_rep: int = 24, seed: int = TEST_SEED + 40_000,
                     thresholds=RT_THRESHOLDS) -> dict:
    """E5 (b): repeat trials after response-cue intervals from 100 ms to 5 s (the trained range ends at 600 ms).

    Returns rt, acc, premature and rt_raw as [S, theta, task, RCI, CSI].
    """
    S = model.n_networks
    csi_levels = [cfg.steps(ms) for ms in E5_LONG_CSI_MS]
    levels = {"context": [TASK_A, TASK_B], "csi": csi_levels, "rel": [-cfg.signal, cfg.signal],
              "irr": [-cfg.signal, 0.0, cfg.signal]}
    rng = torch.Generator().manual_seed(seed)
    per_rci = []
    for ri, rci_ms in enumerate(E5_RCI_MS):
        probe = _probe_grid(cfg, levels, n_rep)
        B = len(probe["context"])
        prime_ab = cfg.signal * (2.0 * torch.randint(0, 2, (S, B, 2), generator=rng) - 1.0)
        seq = _pair_sequences(cfg, S, probe, probe["context"], prime_ab, torch.full((B,), cfg.steps(rci_ms)), seed + 1 + ri)
        y = model.simulate(seq.x, sigma_rec=cfg.sigma_rec, noise=NoiseSource(seed + 2 + ri))
        resp = responses(y, seq, cfg, thresholds)
        ctx, csi = seq.context[0, :, 1], seq.csi[0, :, 1]
        cells = [(ctx == task) & (csi == c) for task in (TASK_A, TASK_B) for c in csi_levels]
        per_rci.append(_probe_cells(resp, seq, cells, thresholds, (2, len(csi_levels))))
    out = {k: np.stack([m[k] for m in per_rci], axis=3) for k in per_rci[0]}       # [S, theta, task, RCI, CSI]
    out.update(rci_ms=np.asarray(E5_RCI_MS), csi_ms=np.asarray(E5_LONG_CSI_MS))
    return out
