"""E0: systems with known answers that are used to validate the fixed-point and linearisation code.

(i)   a hand-built linear line attractor with known direction r and selection vector l, and a softplus
      network of the form of Eq. (1) in which such a slow mode has been planted at a known fixed point;
(ii)  the two-variable decision model of Wong & Wang (2006), whose fixed points are obtained
      independently from the intersections of its nullclines;
(iii) rank-one networks, in which the direction of the selection vector cannot depend on the context
      (Zhang et al., 2025).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .fixedpoints import softplus


# ----------------------------------------------------------------- (i) linear line attractor
@dataclass
class LinearSystem:
    m: np.ndarray        # [n, n] dynamics matrix:  tau dx/dt = M x + B u
    b: np.ndarray        # [n, n_inputs]
    r: np.ndarray        # unit-norm right eigenvector of the slow mode (direction of the line)
    l: np.ndarray        # left eigenvector with l @ r = 1 (selection vector)
    lam: float           # eigenvalue of the slow mode
    eigvals: np.ndarray  # all eigenvalues


def make_line_attractor(n: int = 20, seed: int = 0, lam: float = 0.0, n_inputs: int = 2,
                        n_complex_pairs: int = 4) -> LinearSystem:
    """A non-normal linear system with one slow mode (eigenvalue `lam`) and n - 1 decaying modes.

    M = V L V^-1 with a random, non-orthogonal V; L is block diagonal with the slow eigenvalue,
    real eigenvalues in [-2, -0.3] and complex pairs a +- ib with a in [-2, -0.3].
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, n))
    blocks = np.zeros((n, n))
    blocks[0, 0] = lam
    eig = [complex(lam)]
    i = 1
    for _ in range(n_complex_pairs):
        a, b = -rng.uniform(0.3, 2.0), rng.uniform(0.2, 1.5)
        blocks[i:i + 2, i:i + 2] = [[a, -b], [b, a]]
        eig += [complex(a, b), complex(a, -b)]
        i += 2
    while i < n:
        a = -rng.uniform(0.3, 2.0)
        blocks[i, i] = a
        eig.append(complex(a))
        i += 1
    v_inv = np.linalg.inv(v)
    m = v @ blocks @ v_inv
    scale = np.linalg.norm(v[:, 0])
    return LinearSystem(m=m, b=rng.normal(size=(n, n_inputs)), r=v[:, 0] / scale, l=v_inv[0] * scale,
                        lam=lam, eigvals=np.asarray(eig))


def simulate_linear(sys: LinearSystem, x0: np.ndarray, u: np.ndarray, dt: float = 10.0, tau: float = 100.0) -> np.ndarray:
    """Euler integration of tau dx/dt = M x + B u(t); u has shape [T, n_inputs]. Returns x [T + 1, n]."""
    a = dt / tau
    x = np.empty((len(u) + 1, len(x0)))
    x[0] = x0
    for t in range(len(u)):
        x[t + 1] = x[t] + a * (sys.m @ x[t] + sys.b @ u[t])
    return x


@dataclass
class PlantedRNN:
    w_rec: np.ndarray    # recurrent weights of a softplus network  tau dh/dt = -h + W phi(h) + bias
    bias: np.ndarray
    h_star: np.ndarray   # the fixed point
    w_in: np.ndarray     # input weights [n, n_inputs]
    lin: LinearSystem    # the linearisation that was planted at h_star (in rate coordinates)


def make_planted_rnn(n: int = 40, seed: int = 0, lam: float = -0.02) -> PlantedRNN:
    """A softplus network whose rate-coordinate linearisation at a known fixed point is a given matrix.

    With G = diag(phi'(h*)), M = -I + G W  <=>  W = G^-1 (M + I); the bias is chosen so that h* is a
    fixed point. The input representation directions are b_k = G w_in,k.
    """
    rng = np.random.default_rng(seed + 1000)
    lin = make_line_attractor(n, seed, lam=lam)
    h_star = rng.normal(0.0, 1.0, size=n)
    gains = expit(h_star)
    w_rec = (lin.m + np.eye(n)) / gains[:, None]
    bias = h_star - w_rec @ softplus(h_star)
    w_in = lin.b / gains[:, None]
    return PlantedRNN(w_rec=w_rec, bias=bias, h_star=h_star, w_in=w_in, lin=lin)


# ------------------------------------------------------------------- (ii) Wong & Wang (2006)
@dataclass(frozen=True)
class WongWang:
    """Reduced two-variable model (Wong & Wang, 2006, Appendix): S_i = NMDA gating variable of population i.

        dS_i/dt = -S_i / tau_s + (1 - S_i) gamma H(x_i)
        H(x)    = (a x - b) / (1 - exp(-d (a x - b)))
        x_1     = J_11 S_1 - J_12 S_2 + I_0 + J_ext mu_0 (1 + c / 100)
        x_2     = J_22 S_2 - J_21 S_1 + I_0 + J_ext mu_0 (1 - c / 100)

    Time is in seconds.
    """

    a: float = 270.0          # 1 / (V nC)
    b: float = 108.0          # Hz
    d: float = 0.154          # s
    gamma: float = 0.641
    tau_s: float = 0.1        # s
    j11: float = 0.2609       # nA
    j22: float = 0.2609
    j12: float = 0.0497
    j21: float = 0.0497
    j_ext: float = 5.2e-4     # nA / Hz
    i0: float = 0.3255        # nA
    mu0: float = 30.0         # Hz, stimulus strength
    coh: float = 0.0          # motion coherence in percent

    def rate(self, x):
        z = self.a * np.asarray(x, dtype=np.float64) - self.b
        small = np.abs(z) < 1e-9
        safe = np.where(small, 1.0, z)
        return np.where(small, 1.0 / self.d, safe / (1.0 - np.exp(-self.d * safe)))

    def currents(self, s1, s2):
        x1 = self.j11 * s1 - self.j12 * s2 + self.i0 + self.j_ext * self.mu0 * (1.0 + self.coh / 100.0)
        x2 = self.j22 * s2 - self.j21 * s1 + self.i0 + self.j_ext * self.mu0 * (1.0 - self.coh / 100.0)
        return x1, x2

    def field(self, s):
        """dS/dt for a state s = (S_1, S_2)."""
        s1, s2 = s[..., 0], s[..., 1]
        x1, x2 = self.currents(s1, s2)
        f1 = -s1 / self.tau_s + (1.0 - s1) * self.gamma * self.rate(x1)
        f2 = -s2 / self.tau_s + (1.0 - s2) * self.gamma * self.rate(x2)
        return np.stack([f1, f2], axis=-1)

    def jacobian(self, s, eps: float = 1e-7):
        jac = np.empty((2, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = eps
            jac[:, j] = (self.field(s + step) - self.field(s - step)) / (2 * eps)
        return jac


def _roots_on_grid(func, lo: float, hi: float, n: int) -> list[float]:
    """All sign changes of a (vectorised) scalar function on a grid, refined with Brent's method."""
    grid = np.linspace(lo, hi, n)
    vals = func(grid)
    roots = [float(g) for g in grid[vals == 0.0]]
    for i in np.flatnonzero(vals[:-1] * vals[1:] < 0):
        roots.append(float(brentq(func, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-14)))
    return sorted(roots)


def wong_wang_nullcline_fixed_points(model: WongWang, n_grid: int = 2001) -> np.ndarray:
    """Fixed points from the intersections of the two nullclines (no use of the fixed-point search).

    For every S_1 the S_2-nullcline is solved for S_2 (dS_2/dt = 0, possibly several branches); the fixed
    points are the zeros of dS_1/dt along each branch, located by bisection in S_1.
    """
    def s2_on_nullcline(s1: float) -> list[float]:
        def f2(s2):
            s2 = np.asarray(s2, dtype=np.float64)
            return model.field(np.stack([np.full_like(s2, s1), s2], axis=-1))[..., 1]

        return _roots_on_grid(f2, 0.0, 1.0, 801)

    grid = np.linspace(0.0, 1.0, n_grid)
    branches = [s2_on_nullcline(float(s1)) for s1 in grid]
    points = []
    for i in range(n_grid - 1):
        if len(branches[i]) != len(branches[i + 1]):
            continue                      # a fold of the nullcline; handled by the neighbouring cells
        for j in range(len(branches[i])):
            def f1_on_branch(s1: float, j=j) -> float:
                return float(model.field(np.array([s1, s2_on_nullcline(s1)[j]]))[0])

            fa = float(model.field(np.array([grid[i], branches[i][j]]))[0])
            fb = float(model.field(np.array([grid[i + 1], branches[i + 1][j]]))[0])
            if fa == 0.0:
                points.append((float(grid[i]), branches[i][j]))
            elif fa * fb < 0:
                try:
                    s1 = brentq(f1_on_branch, grid[i], grid[i + 1], xtol=1e-13, rtol=1e-13)
                except (ValueError, IndexError):
                    continue              # the number of branches changed inside the cell
                points.append((float(s1), s2_on_nullcline(float(s1))[j]))
    pts = np.array(sorted(set((round(a, 9), round(b, 9)) for a, b in points)))
    return pts.reshape(-1, 2)


# ------------------------------------------------------------------------ (iii) rank-one networks
def rank_one_modes(m: np.ndarray, n: np.ndarray, gains: np.ndarray):
    """Analytical slow mode of a rank-one network W = m n^T / N at a point with gains G.

    M = -I + G m n^T / N has the eigenvalue -1 + n^T G m / N with right eigenvector G m and left
    eigenvector n; all other eigenvalues are -1. Returns (lam, r, l) with |r| = 1 and l @ r = 1.
    """
    gm = gains * m
    lam = -1.0 + float(n @ gm) / len(m)
    r = gm / np.linalg.norm(gm)
    l = n / float(n @ r)
    return lam, r, l
