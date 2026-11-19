"""The hand-written backward pass must agree with autograd, for every parametrisation of W_rec."""

import pytest
import torch

from stroop_rnn.config import TaskConfig
from stroop_rnn.model import VARIANTS, BatchedRNN
from stroop_rnn.task import make_sequences


def _small_problem(variant, n_networks=3, n_units=12, dtype=torch.float64):
    cfg = TaskConfig()
    gen = torch.Generator().manual_seed(1)
    model = BatchedRNN(seeds=list(range(5, 5 + n_networks)), n_units=n_units, variant=variant).to(dtype)
    seq = make_sequences(cfg, (n_networks, 4), gen, n_trials=2)
    x, target = seq.x.to(dtype), seq.target.to(dtype)
    noise = torch.randn(x.shape[0], n_networks, 4, n_units, generator=gen, dtype=dtype)
    return cfg, model, x, target, noise


@pytest.mark.parametrize("variant", VARIANTS)
def test_bptt_matches_autograd(variant):
    cfg, model, x, target, noise = _small_problem(variant)
    sigma = 0.3

    loss, y = model.bptt(x, target, sigma_rec=sigma, noise=noise)
    manual = {name: p.grad.clone() for name, p in model.named_parameters()}

    model.zero_grad()
    y_ref = model.forward_autograd(x, noise=noise, sigma_rec=sigma)
    loss_ref = (y_ref - target).pow(2).mean(dim=(0, 2, 3))
    loss_ref.sum().backward()

    assert torch.allclose(y, y_ref, atol=1e-10)
    assert torch.allclose(loss, loss_ref, atol=1e-12)
    for name, p in model.named_parameters():
        scale = p.grad.abs().max()
        assert scale > 0, name
        assert torch.allclose(manual[name], p.grad, atol=1e-9 * float(scale), rtol=1e-7), name


@pytest.mark.parametrize("variant", VARIANTS)
def test_simulate_matches_reference(variant):
    cfg, model, x, target, _ = _small_problem(variant)
    y = model.simulate(x)
    y_ref = model.forward_autograd(x).detach()
    assert torch.allclose(y, y_ref, atol=1e-10)


def test_networks_in_a_batch_are_independent():
    """A network's output and gradient do not depend on which other networks share the batch."""
    cfg, model, x, target, noise = _small_problem("full")
    loss, y = model.bptt(x, target, sigma_rec=0.2, noise=noise)
    grad = model.w_raw.grad.clone()

    solo = BatchedRNN(seeds=[model.seeds[1]], n_units=model.n_units).double()
    assert torch.equal(solo.w_raw[0], model.w_raw[1])
    loss1, y1 = solo.bptt(x[:, 1:2], target[:, 1:2], sigma_rec=0.2, noise=noise[:, 1:2])
    assert torch.allclose(loss1[0], loss[1], atol=1e-12)
    assert torch.allclose(y1[:, 0], y[:, 1], atol=1e-12)
    assert torch.allclose(solo.w_raw.grad[0], grad[1], atol=1e-12)


def test_dale_signs_and_rank_one_structure():
    dale = BatchedRNN(seeds=[0, 1], n_units=20, variant="dale")
    w = dale.w_rec()
    n_exc = 16
    assert (w[:, :, :n_exc] >= 0).all() and (w[:, :, n_exc:] <= 0).all()
    radius = torch.linalg.eigvals(w).abs().max(dim=1).values
    assert torch.allclose(radius, torch.full_like(radius, dale.g), atol=1e-4)

    rank1 = BatchedRNN(seeds=[0, 1], n_units=20, variant="rank1")
    assert (torch.linalg.matrix_rank(rank1.w_rec()) == 1).all()


def test_intervention_overwrites_state():
    cfg, model, x, _, _ = _small_problem("full")
    S, B, N = x.shape[1], x.shape[2], model.n_units
    y, h = model.simulate(x, return_h=True)
    t_reset = 30
    mask = torch.zeros(S, B, dtype=torch.bool)
    mask[:, 0] = True
    h_new = torch.zeros(S, B, N, dtype=x.dtype)
    y2, h2 = model.simulate(x, return_h=True, interventions={t_reset: (mask, h_new)})
    # untouched sequences are identical; the reset sequence restarts from zero at t_reset
    assert torch.equal(y2[:, :, 1:], y[:, :, 1:])
    assert torch.equal(y2[:t_reset], y[:t_reset])
    y_fresh = model.simulate(x[t_reset:, :, :1])
    assert torch.allclose(y2[t_reset:, :, :1], y_fresh, atol=1e-12)
