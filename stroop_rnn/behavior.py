"""Behavioural measures (Eq. 6): response times, accuracy, interference, facilitation and switch costs.

The decision variable of a trial is d(t) = y1(t) - y2(t). Responses are detected in a window that
starts at stimulus onset and lasts for the stimulus plus the shortest hold period.

    response       first time |d| exceeds the threshold theta; RT is measured from stimulus onset
    response error that first crossing is towards the wrong output
    premature      |d| already exceeds theta at the last time step before the stimulus: the previous
                   response is still being held, so the "response" is not a response to the stimulus
    final choice   sign of d at the end of the hold period (the choice the network commits to)

RT statistics use correct responses only, as is standard for response-time data. Premature responses
are anticipations: by default they are excluded from response times and response accuracy and their
rate is reported separately (all measures can also be computed without this exclusion).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .config import MAIN_THRESHOLD, RT_THRESHOLDS, TaskConfig
from .task import CONGRUENT, INCONGRUENT, NEUTRAL, TASK_A, TASK_B, Sequences


def decision_trace(y: Tensor, seq: Sequences, cfg: TaskConfig, window: int | None = None) -> Tensor:
    """d = y1 - y2 in the response window of every trial, signed so that positive = correct. [S, B, n, W]"""
    T = y.shape[0]
    W = cfg.window_steps if window is None else window
    d = (y[..., 0] - y[..., 1]).permute(1, 2, 0)                  # [S, B, T]
    idx = seq.stim_on.unsqueeze(-1) + torch.arange(W)
    idx = idx.clamp(max=T - 1)                                      # only matters for truncated (invalid) trials
    n = idx.shape[2]
    trace = d.unsqueeze(2).expand(-1, -1, n, -1).gather(3, idx)     # [S, B, n, W]
    sign = torch.where(seq.choice == 1, -1.0, 1.0)
    return trace * sign.unsqueeze(-1)


def final_decision(y: Tensor, seq: Sequences) -> Tensor:
    """d = y1 - y2 at the last step of the hold period, signed so that positive = correct. [S, B, n]"""
    T = y.shape[0]
    d = (y[..., 0] - y[..., 1]).permute(1, 2, 0)                  # [S, B, T]
    last = (seq.cue_off - 1).clamp(max=T - 1)
    sign = torch.where(seq.choice == 1, -1.0, 1.0)
    return d.gather(2, last) * sign


def pre_stimulus_decision(y: Tensor, seq: Sequences) -> Tensor:
    """|d| at the last time step before the stimulus (0 for a trial that starts at time 0). [S, B, n]"""
    T = y.shape[0]
    d = (y[..., 0] - y[..., 1]).permute(1, 2, 0)                  # [S, B, T]
    before = seq.stim_on - 1
    value = d.gather(2, before.clamp(min=0, max=T - 1)).abs()
    return torch.where(before >= 0, value, torch.zeros_like(value))


def responses(y: Tensor, seq: Sequences, cfg: TaskConfig, thresholds=RT_THRESHOLDS) -> dict:
    """Per-trial response measures, each [S, B, n].

    rt[theta]            response time in ms (NaN if the threshold is never crossed)
    resp_correct[theta]  first crossing was towards the correct output
    premature[theta]     |d| was above the threshold before the stimulus came on
    final_correct        the final choice is correct
    """
    out = {"rt": {}, "resp_correct": {}, "premature": {}, "final_correct": final_decision(y, seq) > 0}
    if not thresholds:
        return out
    trace = decision_trace(y, seq, cfg)
    before = pre_stimulus_decision(y, seq)
    for theta in thresholds:
        crossed = trace.abs() > theta
        any_cross = crossed.any(-1)
        first = crossed.float().argmax(-1)                          # index of the first crossing
        rt = (first + 1).float() * cfg.dt
        out["rt"][theta] = torch.where(any_cross, rt, torch.full_like(rt, float("nan")))
        out["resp_correct"][theta] = any_cross & (trace.gather(-1, first.unsqueeze(-1)).squeeze(-1) > 0)
        out["premature"][theta] = before > theta
    return out


def _mean_where(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of `values` over the last two dimensions (sequences, trials) where mask is true. [S]"""
    w = mask.float()
    total = (torch.nan_to_num(values) * w).sum(dim=(1, 2))
    count = w.sum(dim=(1, 2))
    return torch.where(count > 0, total / count.clamp(min=1), torch.full_like(total, float("nan")))


def cell_means(resp: dict, seq: Sequences, mask: Tensor, theta: float = MAIN_THRESHOLD,
               exclude_premature: bool = True) -> dict:
    """Mean RT of correct responses, response accuracy, rate of premature responses and final-choice
    accuracy within `mask`. Each [S]. RT and response accuracy leave out premature responses."""
    mask = mask & seq.valid
    premature = resp["premature"][theta]
    kept = mask & ~premature if exclude_premature else mask
    ok = resp["resp_correct"][theta]
    return {
        "rt": _mean_where(resp["rt"][theta], kept & ok),
        "resp_acc": _mean_where(ok.float(), kept),
        "premature": _mean_where(premature.float(), mask),
        "final_acc": _mean_where(resp["final_correct"].float(), mask),
        "n": kept.float().sum(dim=(1, 2)),
    }


def stroop_measures(resp: dict, seq: Sequences, theta: float = MAIN_THRESHOLD, mask: Tensor | None = None,
                    exclude_premature: bool = True) -> dict:
    """Interference and facilitation (Eq. 6) for each task; every entry is an array over networks.

    interference = RT_incongruent - RT_neutral,   facilitation = RT_neutral - RT_congruent
    The same differences are reported for response-error rates (err_*).
    """
    base = torch.ones_like(seq.valid) if mask is None else mask
    out = {}
    for task, name in ((TASK_A, "A"), (TASK_B, "B")):
        cells = {
            c: cell_means(resp, seq, base & (seq.context == task) & (seq.congruency == c), theta, exclude_premature)
            for c in (CONGRUENT, NEUTRAL, INCONGRUENT)
        }
        for c, label in ((CONGRUENT, "cong"), (NEUTRAL, "neut"), (INCONGRUENT, "incong")):
            out[f"rt_{label}_{name}"] = cells[c]["rt"].numpy()
            out[f"acc_{label}_{name}"] = cells[c]["resp_acc"].numpy()
            out[f"final_acc_{label}_{name}"] = cells[c]["final_acc"].numpy()
        out[f"interference_{name}"] = out[f"rt_incong_{name}"] - out[f"rt_neut_{name}"]
        out[f"facilitation_{name}"] = out[f"rt_neut_{name}"] - out[f"rt_cong_{name}"]
        out[f"err_interference_{name}"] = out[f"acc_neut_{name}"] - out[f"acc_incong_{name}"]
        out[f"err_facilitation_{name}"] = out[f"acc_cong_{name}"] - out[f"acc_neut_{name}"]
        task_all = cell_means(resp, seq, base & (seq.context == task), theta, exclude_premature)
        out[f"rt_{name}"] = task_all["rt"].numpy()
        out[f"acc_{name}"] = task_all["resp_acc"].numpy()
        out[f"final_acc_{name}"] = task_all["final_acc"].numpy()
        out[f"premature_{name}"] = task_all["premature"].numpy()
    # Asymmetry of interference: how much more the weaker task (B) suffers from the stronger feature (A)
    out["interference_asym"] = out["interference_B"] - out["interference_A"]
    out["err_interference_asym"] = out["err_interference_B"] - out["err_interference_A"]
    return out


def switch_measures(resp: dict, seq: Sequences, theta: float = MAIN_THRESHOLD, mask: Tensor | None = None,
                    exclude_premature: bool = True) -> dict:
    """Switch cost = RT_switch - RT_repeat for each task, and the asymmetry (cost into A minus cost into B)."""
    base = torch.ones_like(seq.valid) if mask is None else mask
    out = {}
    for task, name in ((TASK_A, "A"), (TASK_B, "B")):
        rep = cell_means(resp, seq, base & (seq.context == task) & (seq.switch == 0), theta, exclude_premature)
        swi = cell_means(resp, seq, base & (seq.context == task) & (seq.switch == 1), theta, exclude_premature)
        out[f"premature_repeat_{name}"] = rep["premature"].numpy()
        out[f"premature_switch_{name}"] = swi["premature"].numpy()
        out[f"rt_repeat_{name}"] = rep["rt"].numpy()
        out[f"rt_switch_{name}"] = swi["rt"].numpy()
        out[f"acc_repeat_{name}"] = rep["resp_acc"].numpy()
        out[f"acc_switch_{name}"] = swi["resp_acc"].numpy()
        out[f"switch_cost_{name}"] = out[f"rt_switch_{name}"] - out[f"rt_repeat_{name}"]
        out[f"err_switch_cost_{name}"] = out[f"acc_repeat_{name}"] - out[f"acc_switch_{name}"]
    out["switch_asym"] = out["switch_cost_A"] - out["switch_cost_B"]
    out["err_switch_asym"] = out["err_switch_cost_A"] - out["err_switch_cost_B"]
    return out


def task_accuracy(resp: dict, seq: Sequences) -> tuple[np.ndarray, np.ndarray]:
    """Final-choice accuracy of every network on task A and on task B."""
    fc = resp["final_correct"].float()
    acc_a = _mean_where(fc, seq.valid & (seq.context == TASK_A)).numpy()
    acc_b = _mean_where(fc, seq.valid & (seq.context == TASK_B)).numpy()
    return acc_a, acc_b
