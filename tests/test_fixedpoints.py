"""The fixed-point search, the linearisation and the decomposition on systems with known answers."""

import numpy as np
import pytest

from stroop_rnn.fixedpoints import (
    BELOW_TOL, GenericField, RNNField, decision_mode, decompose_gain_change, effective_dimension,
    find_fixed_points, minimise_speed, rate_linearisation, speed,
)
from stroop_rnn.validation import (
    WongWang, make_line_attractor, make_planted_rnn, rank_one_modes, simulate_linear,
    wong_wang_nullcline_fixed_points,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linear_line_attractor_modes(seed):
    sys = make_line_attractor(n=20, seed=seed)
    mode = decision_mode(sys.m, readout=sys.r)
    assert abs(mode.lam) < 1e-10
    assert np.allclose(mode.r, sys.r, atol=1e-8)
    assert np.allclose(mode.l, sys.l, atol=1e-7 * np.linalg.norm(sys.l))
    assert abs(mode.l @ mode.r - 1.0) < 1e-10
    # the system is non-normal: the selection vector is not the direction of the line
    assert mode.cos_lr < 0.99


def test_linear_line_attractor_fixed_points_lie_on_the_line():
    sys = make_line_attractor(n=20, seed=3)
    field = GenericField(lambda x: sys.m @ x, dim=20, tau=100.0)
    rng = np.random.default_rng(0)
    out = minimise_speed(field, rng.normal(size=(30, 20)), tol=1e-10)
    assert np.all(out.status == BELOW_TOL)
    off_line = out.h - np.outer(out.h @ sys.r, sys.r)
    assert np.abs(off_line).max() < 1e-6
    assert np.ptp(out.h @ sys.r) > 0.5          # a continuum of fixed points, not a single one


def test_linear_dynamics_follow_the_selection_vector():
    """The state relaxes to (l . x0) r, and an input moves it along the line by l . b per unit time (Eq. 4)."""
    sys = make_line_attractor(n=20, seed=4)
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=20)
    x = simulate_linear(sys, x0, np.zeros((6000, 2)))
    assert np.allclose(x[-1], (sys.l @ x0) * sys.r, atol=1e-8)

    dt, tau, n_on = 10.0, 100.0, 30
    for k in range(2):
        u = np.zeros((6000, 2))
        u[:n_on, k] = 0.7
        x = simulate_linear(sys, np.zeros(20), u, dt=dt, tau=tau)
        predicted = (sys.l @ sys.b[:, k]) * 0.7 * n_on * dt / tau
        assert np.allclose(x[-1], predicted * sys.r, atol=1e-8)


@pytest.mark.parametrize("seed", [0, 1])
def test_planted_slow_mode_in_softplus_network(seed):
    net = make_planted_rnn(n=40, seed=seed, lam=-0.02)
    field = RNNField(net.w_rec, net.bias, tau=100.0)
    assert speed(field, net.h_star)[0] < 1e-12
    rng = np.random.default_rng(seed)
    ics = net.h_star + 0.05 * rng.normal(size=(20, 40))
    pts, spd, counts = find_fixed_points(field, ics, tol=1e-11)
    # h* is found. (The nonlinearity can create a second fixed point further along the slow direction,
    # where the weak restoring force of the planted mode is cancelled; that one is a genuine fixed point too.)
    assert counts.sum() == 20 and np.all(spd < 1e-11)
    nearest = int(np.argmin(np.linalg.norm(pts - net.h_star, axis=1)))
    assert np.allclose(pts[nearest], net.h_star, atol=1e-6)

    m, gains = rate_linearisation(net.w_rec, pts[nearest])
    assert np.allclose(m, net.lin.m, atol=1e-6)
    mode = decision_mode(m, readout=net.lin.r)
    assert abs(mode.lam - (-0.02)) < 1e-6
    assert np.allclose(mode.r, net.lin.r, atol=1e-6)
    assert np.allclose(mode.l, net.lin.l, atol=1e-5 * np.linalg.norm(net.lin.l))
    # input representation directions b_k = G w_in,k and effective gains l . b_k
    b = gains[:, None] * net.w_in
    assert np.allclose(b, net.lin.b, atol=1e-6)
    assert np.allclose(mode.l @ b, net.lin.l @ net.lin.b, rtol=1e-5)


@pytest.mark.parametrize("mu0, coh, n_expected, n_stable", [(30.0, 0.0, 3, 2), (0.0, 0.0, 5, 3), (30.0, 6.4, 3, 2)])
def test_wong_wang_fixed_points_match_nullclines(mu0, coh, n_expected, n_stable):
    model = WongWang(mu0=mu0, coh=coh)
    reference = wong_wang_nullcline_fixed_points(model)
    assert len(reference) == n_expected

    field = GenericField(model.field, dim=2, tau=1.0)
    g = np.linspace(0.02, 0.98, 13)
    ics = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    pts, spd, _ = find_fixed_points(field, ics, tol=1e-10, merge_tol=1e-5)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    reference = reference[np.lexsort((reference[:, 1], reference[:, 0]))]
    assert len(pts) == n_expected
    assert np.allclose(pts, reference, atol=1e-6)

    stable = [bool(np.all(np.linalg.eigvals(model.jacobian(p)).real < 0)) for p in pts]
    assert sum(stable) == n_stable


def test_rank_one_selection_vector_direction_is_fixed():
    rng = np.random.default_rng(5)
    n_units = 64
    m = rng.normal(size=n_units)
    n = 2.5 * m + 0.5 * rng.normal(size=n_units)
    w = np.outer(m, n) / n_units
    b = rng.normal(size=n_units)
    modes, inputs = [], []
    for _ in range(2):                                   # two "contexts": two different operating points
        h = rng.normal(size=n_units)
        mat, gains = rate_linearisation(w, h)
        mode = decision_mode(mat, readout=gains * m)
        lam, r, l = rank_one_modes(m, n, gains)
        assert abs(mode.lam - lam) < 1e-10
        assert np.allclose(mode.r, r, atol=1e-8) and np.allclose(mode.l, l, atol=1e-8)
        modes.append(mode)
        inputs.append(gains * b)
    l1, l2 = modes[0].l, modes[1].l
    cos = l1 @ l2 / (np.linalg.norm(l1) * np.linalg.norm(l2))
    assert abs(cos - 1.0) < 1e-10
    # with unit-norm selection vectors the whole change of gain is input modulation
    dec = decompose_gain_change(l1 / np.linalg.norm(l1), inputs[0], l2 / np.linalg.norm(l2), inputs[1])
    assert abs(dec["sel_mod"]) < 1e-10 * abs(dec["inp_mod"])
    assert abs(dec["sel_fraction"]) < 1e-9


def test_decomposition_is_exact():
    rng = np.random.default_rng(6)
    l1, l2, b1, b2 = rng.normal(size=(4, 30))
    dec = decompose_gain_change(l1, b1, l2, b2)
    assert np.isclose(dec["sel_mod"] + dec["inp_mod"], l1 @ b1 - l2 @ b2)
    assert np.isclose(dec["delta_gain"], dec["gain_rel"] - dec["gain_irr"])
    # pure cases
    assert np.isclose(decompose_gain_change(l1, b1, l1, b2)["sel_fraction"], 0.0)
    assert np.isclose(decompose_gain_change(l1, b1, l2, b1)["sel_fraction"], 1.0)


def test_effective_dimension():
    rng = np.random.default_rng(7)
    u, v = rng.normal(size=(2, 50))
    assert np.isclose(effective_dimension(np.outer(u, v)), 1.0)
    assert np.isclose(effective_dimension(np.eye(50)), 50.0)
