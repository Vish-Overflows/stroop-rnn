"""Task generator: Stroop-like context-dependent selection, as single trials or back-to-back sequences.

One trial:

    [ RCI: nothing on ] [ CSI: cue on ] [ stimulus + cue on ] [ hold: cue on, stimulus off ]
                                        |<------------- response window --------------->|

The cue is a one-hot context input that stays on from cue onset until the end of the hold period.
During the response window the target is the one-hot choice given by the sign of the cued feature
(output 0 for a positive sign, output 1 for a negative sign); at all other times the target is zero.
The hold period has a variable, unpredictable duration, so the choice has to be maintained as a
stable state rather than as a timed transient. Trials follow each other without resetting the
network, so the response-stimulus interval is RSI = RCI + CSI.

Sequences are generated for a whole batch of networks at once: all tensors carry leading dimensions
[S, B] (networks, sequences) and time series are laid out time-first, [T, S, B, C].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import TaskConfig

CONGRUENT, NEUTRAL, INCONGRUENT = 0, 1, 2
TASK_A, TASK_B = 0, 1


@dataclass
class Sequences:
    x: Tensor            # [T, S, B, 4]  inputs (u_A, u_B, c_A, c_B)
    target: Tensor       # [T, S, B, 2]
    context: Tensor      # [S, B, n]  0 = task A, 1 = task B
    rel: Tensor          # [S, B, n]  signed value of the cued (relevant) feature
    irr: Tensor          # [S, B, n]  signed value of the other (irrelevant) feature
    choice: Tensor       # [S, B, n]  correct output index (0 or 1); -1 if the relevant feature is zero
    congruency: Tensor   # [S, B, n]  CONGRUENT / NEUTRAL / INCONGRUENT
    switch: Tensor       # [S, B, n]  1 = context differs from the previous trial, 0 = repeat, -1 = first trial
    csi: Tensor          # [S, B, n]  cue-stimulus interval (steps)
    rci: Tensor          # [S, B, n]  response-cue interval (steps)
    hold: Tensor         # [S, B, n]  hold period (steps)
    cue_on: Tensor       # [S, B, n]  first step with the cue on
    stim_on: Tensor      # [S, B, n]  first step with the stimulus on
    stim_off: Tensor     # [S, B, n]  first step after the stimulus
    cue_off: Tensor      # [S, B, n]  first step after the hold period
    valid: Tensor        # [S, B, n]  trial lies completely inside the simulated horizon

    @property
    def n_steps(self) -> int:
        return self.x.shape[0]


def _draw_steps(cfg: TaskConfig, shape, gen, bounds_ms, choices_ms) -> Tensor:
    if choices_ms is not None:
        levels = torch.tensor([cfg.steps(ms) for ms in choices_ms])
        return levels[torch.randint(0, len(levels), shape, generator=gen)]
    lo, hi = cfg.steps(bounds_ms[0]), cfg.steps(bounds_ms[1])
    return torch.randint(lo, hi + 1, shape, generator=gen)


def make_sequences(
    cfg: TaskConfig,
    shape: tuple[int, int],
    gen: torch.Generator,
    *,
    p_a: float = 0.5,
    n_trials: int | None = None,
    horizon: int | None = None,
    csi_choices: tuple[float, ...] | None = None,
    rci_choices: tuple[float, ...] | None = None,
    hold_choices: tuple[float, ...] | None = None,
    context: Tensor | None = None,
    rel: Tensor | None = None,
    irr: Tensor | None = None,
    csi: Tensor | None = None,
    rci: Tensor | None = None,
    hold: Tensor | None = None,
    sigma_in: float | None = None,
    noise: Tensor | None = None,
    tail_steps: int = 10,
) -> Sequences:
    """Sample a batch of trial sequences.

    shape        (S, B): number of networks and of sequences per network.
    p_a          probability that a trial is a task-A trial (contexts are drawn independently per trial,
                 so with p_a = 0.5 the context changes from trial to trial with probability 0.5).
    horizon      fixed number of time steps (longer sequences are cut, shorter ones padded with rest);
                 by default the batch is as long as its longest sequence plus `tail_steps`.
    csi_choices, rci_choices, hold_choices
                 draw the intervals from these discrete levels (ms) instead of the uniform training ranges.
    context, rel, irr, csi, rci, hold
                 optional preset trial variables [S, B, n] (intervals in steps) that override sampling.
    noise        optional preset standard-normal stimulus noise [T, S, B, 2] (instead of drawing it), so
                 that several versions of a sequence can share their noise.
    """
    S, B = shape
    n = cfg.n_trials if n_trials is None else n_trials
    dims = (S, B, n)
    sigma_in = cfg.sigma_in if sigma_in is None else sigma_in

    if context is None:
        context = (torch.rand(dims, generator=gen) >= p_a).long()
    if rel is None:
        rel = cfg.signal * (2.0 * torch.randint(0, 2, dims, generator=gen) - 1.0)
    if irr is None:
        irr = cfg.signal * (torch.randint(0, 3, dims, generator=gen) - 1.0)
    if csi is None:
        csi = _draw_steps(cfg, dims, gen, cfg.csi_ms, csi_choices)
    if rci is None:
        rci = _draw_steps(cfg, dims, gen, cfg.rci_ms, rci_choices)
    if hold is None:
        hold = _draw_steps(cfg, dims, gen, cfg.hold_ms, hold_choices)
    context, csi, rci, hold = context.long(), csi.long(), rci.long(), hold.long()
    rel, irr = rel.float(), irr.float()

    stim = cfg.steps(cfg.stim_ms)
    trial_len = rci + csi + stim + hold
    start = torch.cumsum(trial_len, dim=2) - trial_len
    cue_on = start + rci
    stim_on = cue_on + csi
    stim_off = stim_on + stim
    cue_off = stim_off + hold
    T = int(cue_off.max()) + tail_steps if horizon is None else horizon

    t = torch.arange(T).view(T, 1, 1, 1)
    in_cue = (t >= cue_on) & (t < cue_off)          # [T, S, B, n]
    in_stim = (t >= stim_on) & (t < stim_off)
    in_resp = (t >= stim_on) & (t < cue_off)

    is_a = context == TASK_A
    val_a = torch.where(is_a, rel, irr)
    val_b = torch.where(is_a, irr, rel)
    choice = torch.where(rel > 0, 0, torch.where(rel < 0, 1, -1))

    x = torch.zeros(T, S, B, 4)
    x[..., 0] = (in_stim * val_a).sum(-1)
    x[..., 1] = (in_stim * val_b).sum(-1)
    x[..., 2] = (in_cue & is_a).sum(-1)
    x[..., 3] = (in_cue & ~is_a).sum(-1)
    if sigma_in > 0:
        if noise is None:
            noise = torch.randn(T, S, B, 2, generator=gen)
        x[..., :2] += sigma_in * noise * in_stim.any(-1, keepdim=True)

    target = torch.zeros(T, S, B, 2)
    target[..., 0] = (in_resp & (choice == 0)).sum(-1)
    target[..., 1] = (in_resp & (choice == 1)).sum(-1)

    same_sign = torch.sign(rel) == torch.sign(irr)
    congruency = torch.where(irr == 0, NEUTRAL, torch.where(same_sign, CONGRUENT, INCONGRUENT))
    switch = torch.full(dims, -1, dtype=torch.long)
    switch[..., 1:] = (context[..., 1:] != context[..., :-1]).long()

    return Sequences(
        x=x, target=target, context=context, rel=rel, irr=irr, choice=choice, congruency=congruency,
        switch=switch, csi=csi, rci=rci, hold=hold, cue_on=cue_on, stim_on=stim_on, stim_off=stim_off,
        cue_off=cue_off, valid=cue_off <= T,
    )
