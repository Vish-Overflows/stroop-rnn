"""Task generator and behavioural measures."""

import numpy as np
import torch

from stroop_rnn.behavior import responses, stroop_measures, switch_measures, task_accuracy
from stroop_rnn.config import TaskConfig
from stroop_rnn.task import CONGRUENT, INCONGRUENT, NEUTRAL, TASK_A, TASK_B, make_sequences

CFG = TaskConfig()


def _seq(shape=(2, 50), seed=0, **kw):
    return make_sequences(CFG, shape, torch.Generator().manual_seed(seed), **kw)


def test_trial_layout_and_targets():
    seq = _seq(sigma_in=0.0)
    S, B, n = seq.context.shape
    T = seq.n_steps
    assert n == CFG.n_trials and seq.x.shape == (T, S, B, 4) and seq.target.shape == (T, S, B, 2)
    stim = CFG.steps(CFG.stim_ms)
    for s, b in [(0, 0), (1, 7), (0, 31)]:
        expected_x = torch.zeros(T, 4)
        expected_y = torch.zeros(T, 2)
        t = 0
        for i in range(n):
            rci, csi, hold = int(seq.rci[s, b, i]), int(seq.csi[s, b, i]), int(seq.hold[s, b, i])
            cue_on, stim_on = t + rci, t + rci + csi
            stim_off, cue_off = stim_on + stim, stim_on + stim + hold
            assert (int(seq.cue_on[s, b, i]), int(seq.stim_on[s, b, i])) == (cue_on, stim_on)
            assert (int(seq.stim_off[s, b, i]), int(seq.cue_off[s, b, i])) == (stim_off, cue_off)
            ctx = int(seq.context[s, b, i])
            expected_x[cue_on:cue_off, 2 + ctx] = 1.0
            rel, irr = float(seq.rel[s, b, i]), float(seq.irr[s, b, i])
            expected_x[stim_on:stim_off, ctx] = rel            # the cued feature
            expected_x[stim_on:stim_off, 1 - ctx] = irr        # the other feature
            expected_y[stim_on:cue_off, 0 if rel > 0 else 1] = 1.0
            assert int(seq.choice[s, b, i]) == (0 if rel > 0 else 1)
            t = cue_off
        assert torch.equal(seq.x[:, s, b], expected_x)
        assert torch.equal(seq.target[:, s, b], expected_y)


def test_intervals_and_labels():
    seq = _seq(shape=(3, 400), seed=1)
    lo, hi = (CFG.steps(v) for v in CFG.csi_ms)
    assert int(seq.csi.min()) == lo and int(seq.csi.max()) == hi
    lo, hi = (CFG.steps(v) for v in CFG.rci_ms)
    assert int(seq.rci.min()) == lo and int(seq.rci.max()) == hi
    lo, hi = (CFG.steps(v) for v in CFG.hold_ms)
    assert int(seq.hold.min()) == lo and int(seq.hold.max()) == hi
    assert set(seq.rel.unique().tolist()) == {-1.0, 1.0}
    assert set(seq.irr.unique().tolist()) == {-1.0, 0.0, 1.0}
    # congruency
    assert torch.all((seq.congruency == NEUTRAL) == (seq.irr == 0))
    assert torch.all((seq.congruency == CONGRUENT) == (seq.irr == seq.rel))
    assert torch.all((seq.congruency == INCONGRUENT) == (seq.irr == -seq.rel))
    # transitions
    assert torch.all(seq.switch[..., 0] == -1)
    assert torch.equal(seq.switch[..., 1:], (seq.context[..., 1:] != seq.context[..., :-1]).long())
    # the context changes with probability 0.5 when both tasks are equally likely
    assert abs(float(seq.switch[..., 1:].float().mean()) - 0.5) < 0.03
    assert abs(float((seq.context == TASK_A).float().mean()) - 0.5) < 0.03


def test_task_ratio_and_discrete_levels():
    seq = _seq(shape=(2, 500), seed=2, p_a=0.8, csi_choices=(100, 800), rci_choices=(200,), hold_choices=(600,))
    assert abs(float((seq.context == TASK_A).float().mean()) - 0.8) < 0.02
    assert set(seq.csi.unique().tolist()) == {10, 80}
    assert set(seq.rci.unique().tolist()) == {20} and set(seq.hold.unique().tolist()) == {60}
    only_a = _seq(seed=3, p_a=1.0)
    assert torch.all(only_a.context == TASK_A)


def test_noise_only_during_the_stimulus_and_horizon():
    cfg = TaskConfig()
    seq = make_sequences(cfg, (2, 60), torch.Generator().manual_seed(4), horizon=700)
    assert seq.n_steps == 700
    assert torch.equal(seq.valid, seq.cue_off <= 700) and not bool(seq.valid.all()) and bool(seq.valid[..., 0].all())
    t = torch.arange(700).view(-1, 1, 1, 1)
    in_stim = ((t >= seq.stim_on) & (t < seq.stim_off)).any(-1)          # [T, S, B]
    assert torch.all(seq.x[..., :2][~in_stim] == 0)
    noisy = seq.x[..., :2][in_stim]
    assert 2.5 < float(noisy.std()) < 3.5                                # sigma_in = 3 plus the signal


def test_response_times_from_known_outputs():
    """Outputs that ramp at a known rate must give the corresponding response times."""
    seq = _seq(shape=(1, 40), seed=5, sigma_in=0.0)
    T = seq.n_steps
    S, B, n = seq.context.shape
    y = torch.zeros(T, S, B, 2)
    slope = torch.empty(S, B, n).uniform_(0.01, 0.2, generator=torch.Generator().manual_seed(6))
    wrong = torch.rand(S, B, n, generator=torch.Generator().manual_seed(7)) < 0.2
    for b in range(B):
        for i in range(n):
            on, off = int(seq.stim_on[0, b, i]), int(seq.cue_off[0, b, i])
            ramp = (slope[0, b, i] * torch.arange(1, off - on + 1)).clamp(max=1.0)
            out = int(seq.choice[0, b, i])
            if wrong[0, b, i]:
                out = 1 - out
            y[on:off, 0, b, out] = ramp
    resp = responses(y, seq, CFG)
    for theta in (0.3, 0.5, 0.7):
        steps = torch.floor(theta / slope + 1e-6) + 1                    # first step with ramp > theta
        assert torch.allclose(resp["rt"][theta], steps * CFG.dt)
        assert torch.equal(resp["resp_correct"][theta], ~wrong)
    assert torch.equal(resp["final_correct"], ~wrong)
    acc_a, acc_b = task_accuracy(resp, seq)
    is_a = seq.context == TASK_A
    assert np.isclose(acc_a[0], float((~wrong)[is_a].float().mean()))
    assert np.isclose(acc_b[0], float((~wrong)[~is_a].float().mean()))


def test_interference_and_switch_costs_from_known_response_times():
    """Trial-wise RTs constructed with known effects are recovered by the summary measures."""
    seq = _seq(shape=(2, 3000), seed=8, sigma_in=0.0)
    base = {TASK_A: 300.0, TASK_B: 400.0}
    interference = {TASK_A: 20.0, TASK_B: 80.0}
    facilitation = {TASK_A: 10.0, TASK_B: 30.0}
    switch_cost = {TASK_A: 50.0, TASK_B: 20.0}
    rt = torch.zeros(seq.context.shape)
    for task in (TASK_A, TASK_B):
        m = seq.context == task
        rt[m] = base[task]
        rt[m & (seq.congruency == INCONGRUENT)] += interference[task]
        rt[m & (seq.congruency == CONGRUENT)] -= facilitation[task]
        rt[m & (seq.switch == 1)] += switch_cost[task]
    ok = torch.ones_like(seq.valid)
    resp = {"rt": {0.5: rt}, "resp_correct": {0.5: ok}, "premature": {0.5: ~ok}, "final_correct": ok}
    st = stroop_measures(resp, seq, 0.5, mask=seq.switch == 0)           # within repeat trials: exact
    sw = switch_measures(resp, seq, 0.5, mask=seq.congruency == NEUTRAL)
    for task, name in ((TASK_A, "A"), (TASK_B, "B")):
        assert np.allclose(st[f"interference_{name}"], interference[task], atol=1e-3)
        assert np.allclose(st[f"facilitation_{name}"], facilitation[task], atol=1e-3)
        assert np.allclose(sw[f"switch_cost_{name}"], switch_cost[task], atol=1e-3)
    assert np.allclose(st["interference_asym"], 60.0, atol=1e-3)
    assert np.allclose(sw["switch_asym"], 30.0, atol=1e-3)

    # premature responses are left out of the response times unless asked otherwise
    premature = (seq.switch == 0) & (seq.rci <= 20)
    fast = torch.where(premature, torch.full_like(rt, 10.0), rt)
    resp = {"rt": {0.5: fast}, "resp_correct": {0.5: ok}, "premature": {0.5: premature}, "final_correct": ok}
    sw = switch_measures(resp, seq, 0.5, mask=seq.congruency == NEUTRAL)
    raw = switch_measures(resp, seq, 0.5, mask=seq.congruency == NEUTRAL, exclude_premature=False)
    for task, name in ((TASK_A, "A"), (TASK_B, "B")):
        assert np.allclose(sw[f"switch_cost_{name}"], switch_cost[task], atol=1e-3)
        assert np.all(raw[f"switch_cost_{name}"] > switch_cost[task] + 20)
        assert np.all(sw[f"premature_repeat_{name}"] > 0.1) and np.all(sw[f"premature_switch_{name}"] == 0)


def test_premature_responses_are_detected():
    """A response that is still held when the next stimulus arrives counts as premature, not as a fast response."""
    seq = _seq(shape=(1, 30), seed=9, sigma_in=0.0, rci_choices=(100,), csi_choices=(100,))
    T = seq.n_steps
    y = torch.zeros(T, 1, 30, 2)
    held = torch.rand(1, 30, CFG.n_trials, generator=torch.Generator().manual_seed(10)) < 0.5
    held[..., 0] = False
    for b in range(30):
        for i in range(CFG.n_trials):
            on, off = int(seq.stim_on[0, b, i]), int(seq.cue_off[0, b, i])
            y[on + 5:off, 0, b, int(seq.choice[0, b, i])] = 1.0          # response 60 ms after stimulus onset
            if held[0, b, i]:                                             # previous response held through the gap
                y[int(seq.cue_off[0, b, i - 1]):on + 5, 0, b, int(seq.choice[0, b, i - 1])] = 0.6
    resp = responses(y, seq, CFG)
    assert torch.equal(resp["premature"][0.5], held)
    assert not bool(resp["premature"][0.7].any())
    assert torch.all(resp["rt"][0.5][~held] == 60.0) and torch.all(resp["rt"][0.5][held] == 10.0)


def test_battery_versions_pair_every_trial_across_conditions():
    from stroop_rnn.experiments import battery_versions

    versions = list(battery_versions(CFG, (2, 40), torch.Generator().manual_seed(11)))
    assert len(versions) == 9
    base = versions[0]
    cong = torch.stack([v.congruency for v in versions])               # [9, S, B, n]
    ctx = torch.stack([v.context for v in versions])
    sw = torch.stack([v.switch for v in versions])
    for v in versions:
        assert v.n_steps == base.n_steps
        for name in ("rel", "csi", "rci", "hold", "stim_on", "cue_off"):
            assert torch.equal(getattr(v, name), getattr(base, name))
    # rotations: within a flip pattern every trial is congruent, neutral and incongruent exactly once
    for flip in range(3):
        block = cong[3 * flip:3 * flip + 3]
        assert torch.equal(block.sort(dim=0).values, torch.tensor([0, 1, 2]).view(3, 1, 1, 1).expand_as(block))
        assert torch.equal(ctx[3 * flip], ctx[3 * flip + 1]) and torch.equal(ctx[3 * flip], ctx[3 * flip + 2])
    # flips: every transition occurs as a switch and as a repeat with the same current task
    even, odd = torch.arange(1, 6, 2), torch.arange(2, 6, 2)           # 2nd/4th/6th and 3rd/5th trials
    assert torch.equal(ctx[0][..., even], ctx[3][..., even]) and torch.equal(sw[0][..., even], 1 - sw[3][..., even])
    assert torch.equal(ctx[0][..., odd], ctx[6][..., odd]) and torch.equal(sw[0][..., odd], 1 - sw[6][..., odd])
    # the same noise in every version: stimulus inputs differ only by the noise-free signal
    quiet = list(battery_versions(TaskConfig(sigma_in=0.0), (2, 40), torch.Generator().manual_seed(11)))
    noise0 = versions[0].x[..., :2] - quiet[0].x[..., :2]
    for v, q in zip(versions[:3], quiet[:3]):                           # same contexts, rotated irrelevant feature
        assert torch.allclose(v.x[..., :2] - q.x[..., :2], noise0, atol=1e-6)
    assert float(noise0.abs().max()) > 1.0
