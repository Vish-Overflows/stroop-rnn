"""Statistical tools of the analysis plan (Section 7 of the proposal).

- bootstrap confidence intervals over networks
- logistic regression of choice on the two features (with a parameter-recovery test on simulated agents)
- sign-flip and label permutation tests
- thin wrappers around the regression, Wilcoxon and mixed-model routines of SciPy / statsmodels
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps
from scipy.special import expit


# ------------------------------------------------------------------------------- bootstrap
def bootstrap_ci(values, stat=np.mean, n_boot: int = 10000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap confidence interval of a statistic over networks.

    values: [n] or [n, ...] (resampled along the first axis). Returns (estimate, lower, upper).
    NaN entries (excluded networks) are dropped for one-dimensional input.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    if values.ndim == 1 and stat is np.mean:
        boot = values[idx].mean(axis=1)
    else:
        boot = np.array([stat(values[i]) for i in idx])
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha], axis=0)
    return stat(values), lo, hi


# --------------------------------------------------------------------- logistic regression
def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 0.0, max_iter: int = 100, tol: float = 1e-10):
    """Maximum-likelihood logistic regression by Newton's method (iteratively reweighted least squares).

    x: [n, p] regressors (an intercept column is added), y: [n] in {0, 1}.
    Returns (beta [p + 1] with the intercept first, standard errors [p + 1]).
    `ridge` adds a small quadratic penalty on the slopes, which keeps the fit finite when the data are
    (almost) perfectly separable.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    p = design.shape[1]
    penalty = ridge * np.eye(p)
    penalty[0, 0] = 0.0
    beta = np.zeros(p)
    for _ in range(max_iter):
        mu = expit(design @ beta)
        w = mu * (1.0 - mu)
        grad = design.T @ (y - mu) - penalty @ beta
        hess = (design * w[:, None]).T @ design + penalty
        step = np.linalg.solve(hess + 1e-12 * np.eye(p), grad)
        # damp steps that would overshoot (can happen for nearly separable data)
        norm = np.linalg.norm(step)
        if norm > 10.0:
            step *= 10.0 / norm
        beta = beta + step
        if np.linalg.norm(step) < tol * (1.0 + np.linalg.norm(beta)):
            break
    mu = expit(design @ beta)
    hess = (design * (mu * (1.0 - mu))[:, None]).T @ design + penalty
    se = np.sqrt(np.diag(np.linalg.inv(hess + 1e-12 * np.eye(p))))
    return beta, se


def simulate_logistic_agent(beta: np.ndarray, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Choices of an agent whose choice probability is logistic in the regressors x. beta = (b0, b1, ...)."""
    prob = expit(beta[0] + x @ beta[1:])
    return (rng.random(len(x)) < prob).astype(np.float64)


def parameter_recovery(x: np.ndarray, n_agents: int = 200, seed: int = 0, ranges=((-0.5, 0.5), (0.5, 8.0), (-2.0, 2.0)),
                       ridge: float = 0.0) -> dict:
    """Simulate agents with known weights on the design x [n, 2] and refit them.

    ranges: uniform ranges of (bias, weight of feature 1, weight of feature 2).
    Returns the true and recovered weights [n_agents, 3] and summary statistics per weight.
    """
    rng = np.random.default_rng(seed)
    true = np.column_stack([rng.uniform(lo, hi, size=n_agents) for lo, hi in ranges])
    fitted = np.empty_like(true)
    se = np.empty_like(true)
    for i in range(n_agents):
        y = simulate_logistic_agent(true[i], x, rng)
        fitted[i], se[i] = fit_logistic(x, y, ridge=ridge)
    err = fitted - true
    summary = {
        "corr": [float(np.corrcoef(true[:, j], fitted[:, j])[0, 1]) for j in range(true.shape[1])],
        "bias": err.mean(axis=0).tolist(),
        "rmse": np.sqrt((err**2).mean(axis=0)).tolist(),
        "slope": [float(np.polyfit(true[:, j], fitted[:, j], 1)[0]) for j in range(true.shape[1])],
        "coverage95": (np.abs(err) < 1.96 * se).mean(axis=0).tolist(),
    }
    return {"true": true, "fitted": fitted, "se": se, "summary": summary}


# ------------------------------------------------------------------------- permutation tests
def sign_flip_test(diff, n_perm: int = 20000, seed: int = 0) -> dict:
    """Paired permutation test of H0: the within-network differences are symmetric around zero.

    The sign of every network's difference is flipped at random; the test statistic is the mean.
    """
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n == 0:
        return {"mean": float("nan"), "p": float("nan"), "n": 0}
    observed = diff.mean()
    if n <= 16:                                            # exact: all 2^n sign patterns
        signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * n, indexing="ij")).reshape(n, -1).T
        null = (signs * diff).mean(axis=1)
        p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        null = (signs * diff).mean(axis=1)
        p = float((np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (n_perm + 1))
    return {"mean": float(observed), "p": p, "n": int(n)}


def two_sample_permutation_test(a, b, n_perm: int = 20000, seed: int = 0) -> dict:
    """Permutation test of the difference of means between two independent groups of networks."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        null[i] = perm[: len(a)].mean() - perm[len(a):].mean()
    p = float((np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (n_perm + 1))
    return {"diff": float(observed), "p": p, "n_a": int(len(a)), "n_b": int(len(b))}


# -------------------------------------------------------------------------------- wrappers
def wilcoxon_paired(a, b) -> dict:
    """Wilcoxon signed-rank test of paired samples (networks with a NaN in either sample are dropped)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.allclose(a, b):
        return {"statistic": float("nan"), "p": float("nan"), "n": int(len(a)), "median_diff": float("nan")}
    res = sps.wilcoxon(a, b)
    return {"statistic": float(res.statistic), "p": float(res.pvalue), "n": int(len(a)),
            "median_diff": float(np.median(a - b)), "mean_diff": float(np.mean(a - b))}
