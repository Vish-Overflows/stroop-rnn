"""Fixed points, slow points and linearisation (Sussillo & Barak, 2013; Mante et al., 2013).

The search minimises q(h) = 1/2 ||F(h)||^2 (Eq. 2) from many initial conditions with a batched
Levenberg-Marquardt iteration. Everything here is plain NumPy in float64 and works on any vector
field that provides a residual and a Jacobian, so that exactly the same code can be validated on
systems with known answers (E0) before it is applied to the trained networks.

Units: time is measured in ms, so F = (right-hand side of Eq. 1) / tau has units 1/ms. The solver works
with the dimensionless residual R(h) = tau F(h); speeds are reported as ||F|| in 1/ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.special import expit


def softplus(h: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, h)


# --------------------------------------------------------------------------- vector fields
class RNNField:
    """Autonomous dynamics of one network for a constant input:  tau dh/dt = -h + W phi(h) + b."""

    def __init__(self, w_rec: np.ndarray, bias: np.ndarray, tau: float = 100.0):
        self.w = np.asarray(w_rec, dtype=np.float64)
        self.b = np.asarray(bias, dtype=np.float64)
        self.tau = float(tau)
        self.dim = self.w.shape[0]

    def residual(self, h: np.ndarray) -> np.ndarray:
        """R(h) = tau F(h) for a batch of states [K, N]."""
        return -h + softplus(h) @ self.w.T + self.b

    def jacobian(self, h: np.ndarray) -> np.ndarray:
        """dR/dh = -I + W diag(phi'(h)) for a batch of states: [K, N, N]."""
        jac = self.w[None, :, :] * expit(h)[:, None, :]
        idx = np.arange(self.dim)
        jac[:, idx, idx] -= 1.0
        return jac


class GenericField:
    """A vector field given by a function f(h) -> tau dh/dt; the Jacobian is taken by central differences."""

    def __init__(self, func, dim: int, tau: float = 1.0, eps: float = 1e-6):
        self.func, self.dim, self.tau, self.eps = func, dim, float(tau), eps

    def residual(self, h: np.ndarray) -> np.ndarray:
        return np.stack([np.asarray(self.func(row), dtype=np.float64) for row in h])

    def jacobian(self, h: np.ndarray) -> np.ndarray:
        K, N = h.shape
        jac = np.empty((K, N, N))
        for j in range(N):
            step = np.zeros(N)
            step[j] = self.eps
            jac[:, :, j] = (self.residual(h + step) - self.residual(h - step)) / (2 * self.eps)
        return jac


def speed(field, h: np.ndarray) -> np.ndarray:
    """||F(h)|| in units of 1/time for each state."""
    return np.linalg.norm(field.residual(np.atleast_2d(h)), axis=-1) / field.tau


# ------------------------------------------------------------------------------ the search
BELOW_TOL, LOCAL_MIN, MAX_ITER = 0, 1, 2


@dataclass
class SearchResult:
    h: np.ndarray        # [K, N] end points
    speed: np.ndarray    # [K]    ||F|| at the end points
    status: np.ndarray   # [K]    BELOW_TOL, LOCAL_MIN (minimum of q with ||F|| above tol) or MAX_ITER
    n_iter: int


def minimise_speed(field, h0: np.ndarray, tol: float, max_iter: int = 400, mu_init: float = 1e-2,
                   mu_max: float = 1e10, ftol: float = 1e-13) -> SearchResult:
    """Minimise q(h) = 1/2 ||F(h)||^2 from every row of h0 (Levenberg-Marquardt).

    A point stops as soon as ||F|| < tol. Points that reach a minimum of q without meeting the
    tolerance are flagged LOCAL_MIN: either the damping grows without any further decrease of q, or
    three accepted steps in a row lower q by less than a fraction `ftol`.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    K, N = h.shape
    res = field.residual(h)
    q = 0.5 * np.einsum("ki,ki->k", res, res)
    q_tol = 0.5 * (tol * field.tau) ** 2
    mu = np.full(K, mu_init)
    status = np.full(K, MAX_ITER)
    status[q < q_tol] = BELOW_TOL
    flat = np.zeros(K, dtype=int)            # consecutive accepted steps without real progress
    eye = np.eye(N)
    it = 0
    for it in range(1, max_iter + 1):
        act = np.flatnonzero(status == MAX_ITER)
        if act.size == 0:
            break
        jac = field.jacobian(h[act])
        grad = np.einsum("kij,ki->kj", jac, res[act])
        hess = np.matmul(jac.transpose(0, 2, 1), jac) + mu[act, None, None] * eye
        step = -np.linalg.solve(hess, grad[:, :, None])[:, :, 0]
        h_new = h[act] + step
        res_new = field.residual(h_new)
        q_new = 0.5 * np.einsum("ki,ki->k", res_new, res_new)
        better = q_new < q[act]
        acc = act[better]
        small = (q[acc] - q_new[better]) < ftol * q[acc]
        flat[acc] = np.where(small, flat[acc] + 1, 0)
        h[acc], res[acc], q[acc] = h_new[better], res_new[better], q_new[better]
        mu[acc] = np.maximum(mu[acc] / 3.0, 1e-14)
        mu[act[~better]] *= 4.0
        status[act[q[act] < q_tol]] = BELOW_TOL
        stuck = ((mu[act] > mu_max) | (flat[act] >= 3)) & (status[act] == MAX_ITER)
        status[act[stuck]] = LOCAL_MIN
    return SearchResult(h=h, speed=np.sqrt(2.0 * q) / field.tau, status=status, n_iter=it)


class ReadoutConstraint:
    """A network field with one extra residual, sqrt(kappa) (w . phi(h) - target).

    Minimising q for this augmented field gives the slowest state with a prescribed value of a linear
    readout of the rates. With the decision variable as the readout, this traces the floor of the
    valley of q along the decision axis: the slow points of an approximate line attractor.
    """

    def __init__(self, field: RNNField, w: np.ndarray, target: float, kappa: float = 1e4):
        self.field, self.w, self.target = field, np.asarray(w, dtype=np.float64), float(target)
        self.root_kappa = float(np.sqrt(kappa))
        self.tau, self.dim = field.tau, field.dim

    def residual(self, h: np.ndarray) -> np.ndarray:
        extra = self.root_kappa * (softplus(h) @ self.w - self.target)
        return np.concatenate([self.field.residual(h), extra[:, None]], axis=1)

    def jacobian(self, h: np.ndarray) -> np.ndarray:
        extra = self.root_kappa * (expit(h) * self.w)
        return np.concatenate([self.field.jacobian(h), extra[:, None, :]], axis=1)


def slowest_point_with_readout(field: RNNField, w: np.ndarray, target: float, h0: np.ndarray,
                               kappa: float = 1e4, max_iter: int = 400):
    """Minimise q subject to w . phi(h) = target, starting from h0 [N]. Returns (h, ||F||, readout)."""
    out = minimise_speed(ReadoutConstraint(field, w, target, kappa), np.atleast_2d(h0), tol=0.0, max_iter=max_iter)
    h = out.h[0]
    return h, float(speed(field, h)[0]), float(softplus(h) @ w)


def unique_points(h: np.ndarray, tol: float) -> np.ndarray:
    """Indices of a subset of rows such that no two kept points are closer than tol (greedy)."""
    keep: list[int] = []
    for i in range(len(h)):
        if all(np.linalg.norm(h[i] - h[j]) > tol for j in keep):
            keep.append(i)
    return np.asarray(keep, dtype=int)


def find_fixed_points(field, h0: np.ndarray, tol: float = 1e-9, merge_tol: float = 1e-4, **kw):
    """Converge every initial condition to a fixed point and return the distinct ones.

    Returns (points [P, N], speeds [P], multiplicity [P]); only points with ||F|| < tol are returned.
    """
    out = minimise_speed(field, h0, tol=tol, **kw)
    ok = out.status == BELOW_TOL
    pts, spd = out.h[ok], out.speed[ok]
    if len(pts) == 0:
        return pts, spd, np.zeros(0, dtype=int)
    idx = unique_points(pts, merge_tol * max(1.0, np.abs(pts).max()))
    counts = np.array([np.sum(np.linalg.norm(pts - pts[i], axis=1) <= merge_tol * max(1.0, np.abs(pts).max())) for i in idx])
    return pts[idx], spd[idx], counts


# --------------------------------------------------------------------------- linearisation
def rate_linearisation(w_rec: np.ndarray, h_star: np.ndarray):
    """Linearised dynamics in firing-rate coordinates (Eq. 3):  tau d(dr)/dt = M dr + sum_k b_k u_k.

    Returns (M, gains) with M = -I + G W_rec and gains = phi'(h*), so that b_k = gains * w_in,k.
    """
    gains = expit(np.asarray(h_star, dtype=np.float64))
    m = gains[:, None] * np.asarray(w_rec, dtype=np.float64)
    m[np.diag_indices_from(m)] -= 1.0
    return m, gains


@dataclass
class Mode:
    lam: float            # eigenvalue of M (dimensionless: lambda * tau)
    r: np.ndarray         # right eigenvector, unit norm: direction of the line attractor
    l: np.ndarray         # left eigenvector with l @ r = 1: the selection vector
    eigvals: np.ndarray   # all eigenvalues of M
    cond: float           # condition number of the eigenvector matrix (non-normality)
    cos_lr: float         # cosine of the angle between l and r
    second: float = float("nan")     # largest real part among all other eigenvalues
    is_slowest: bool = True          # no other eigenvalue has a larger real part
    overlap: float = 1.0             # |r . r_ref| when the mode was selected by continuity
    index: int = -1                  # position of the mode in eigvals
    left: np.ndarray | None = None   # all left eigenvectors as rows (left @ right = identity)
    right: np.ndarray | None = None  # all right eigenvectors as columns


def decision_mode(m: np.ndarray, readout: np.ndarray | None = None, r_ref: np.ndarray | None = None) -> Mode:
    """The slowest mode of M and its selection vector.

    On a line attractor every other mode decays, so the mode with the largest real eigenvalue is the one
    whose eigenvalue is closest to zero. The same rule picks the unstable mode of a decision saddle,
    which is the fallback when no slow mode has formed. Following Mante et al. (2013) the right
    eigenvector r has unit norm and the left eigenvector l is scaled so that l @ r = 1; the sign is
    chosen so that the readout difference y1 - y2 increases along r.

    If r_ref is given, the mode is selected by continuity instead: the real mode whose right
    eigenvector is best aligned with r_ref. This is used to follow one mode along a set of slow points.
    """
    vals, vr = np.linalg.eig(m)
    left = np.linalg.inv(vr)                 # rows are left eigenvectors, left @ vr = identity
    is_real = np.abs(vals.imag) < 1e-9 * (1.0 + np.abs(vals.real))
    cand = np.flatnonzero(is_real)
    if cand.size == 0:
        raise RuntimeError("the linearisation has no real eigenvalue")
    if r_ref is None:
        i = cand[np.argmax(vals.real[cand])]
        overlap = 1.0
    else:
        cos = np.abs(vr[:, cand].real.T @ r_ref) / np.linalg.norm(vr[:, cand].real, axis=0)
        i = cand[np.argmax(cos)]
        overlap = float(cos.max() / np.linalg.norm(r_ref))
    r = vr[:, i].real
    l = left[i].real
    scale = np.linalg.norm(r)
    r, l = r / scale, l * scale              # keeps l @ r = 1
    flip = (readout @ r < 0) if readout is not None else (r_ref is not None and r_ref @ r < 0)
    if flip:
        r, l = -r, -l
    rest = np.delete(vals.real, i)
    second = float(rest.max()) if rest.size else float("-inf")
    return Mode(lam=float(vals[i].real), r=r, l=l, eigvals=vals, cond=float(np.linalg.cond(vr)),
                cos_lr=float(1.0 / np.linalg.norm(l)), second=second,
                is_slowest=bool(vals[i].real >= second - 1e-12), overlap=overlap, index=int(i),
                left=left, right=vr)


def static_gain_share(mode: Mode, readout: np.ndarray, b: np.ndarray) -> float:
    """Share of the static gain from an input direction b to a readout that is carried by `mode`.

    For stable linear dynamics the steady-state response to a constant input is
        readout . (-M)^-1 b = sum_i (readout . r_i)(l_i . b) / (-lambda_i),
    and the function returns the term of the selected mode divided by the whole sum. A value close
    to one means that the readout responds to this input almost exclusively through that mode.
    """
    terms = (readout @ mode.right) * (mode.left @ b) / (-mode.eigvals)
    total = terms.sum().real
    return float(terms[mode.index].real / total) if total != 0 else float("nan")


# ---------------------------------------------------------------------------- decomposition
def decompose_gain_change(l_rel: np.ndarray, b_rel: np.ndarray, l_irr: np.ndarray, b_irr: np.ndarray) -> dict:
    """Exact split of the context-dependent change of a feature's effective gain (Eq. 5).

        g_rel - g_irr = l_rel.b_rel - l_irr.b_irr = (l_rel - l_irr).(b_rel + b_irr)/2 + (l_rel + l_irr)/2.(b_rel - b_irr)
                                                     selection-vector modulation       input modulation

    "rel"/"irr" are the contexts in which the feature is relevant/irrelevant.
    """
    sel = float((l_rel - l_irr) @ (b_rel + b_irr) / 2.0)
    inp = float((l_rel + l_irr) @ (b_rel - b_irr) / 2.0)
    total = sel + inp
    return {
        "gain_rel": float(l_rel @ b_rel), "gain_irr": float(l_irr @ b_irr), "delta_gain": total,
        "sel_mod": sel, "inp_mod": inp, "sel_fraction": sel / total if total != 0 else float("nan"),
    }


def effective_dimension(w: np.ndarray) -> float:
    """Stable rank of the connectivity, sum_i sigma_i^2 / sigma_1^2 (Zhang et al., 2025)."""
    sv = np.linalg.svd(np.asarray(w, dtype=np.float64), compute_uv=False)
    return float(np.sum(sv**2) / sv[0] ** 2)
