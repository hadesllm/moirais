# morie.fn — function file (hadesllm/morie)
"""Average-derivative estimator (Powell-Stock-Stoker 1989; Horowitz 2009, Ch 4).

Density-weighted average derivative:

    delta = E[ grad m(X) f(X) ] / E[ f(X) ]   approx by
    delta_hat = - (2/n) sum_i Y_i * (grad f_hat(X_i) / f_hat(X_i)) / 1   (Stoker form)

Implementation: use the score-style estimator
    delta_hat = - (1/n) sum_i 2 Y_i * (1/h^2) * sum_{j != i} (X_i - X_j) K_h(X_i-X_j) / (n-1)
which is the standard Powell-Stock-Stoker direct estimator with a
Gaussian kernel.  SE from the asymptotic IID influence function.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_average_derivative"]


def _silverman(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if sigma <= 0:
        sigma = max(s, 1e-6)
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def horowitz_average_derivative(x, y, bandwidth=None):
    """Powell-Stock-Stoker density-weighted average-derivative estimator.

    Returns the p-dimensional average derivative ``delta_hat`` together
    with its IID-influence-function-based SE.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n, p = X.shape
    if n < max(20, 2 * p):
        return RichResult(payload={"estimate": np.full(p, np.nan),
                                   "se": np.full(p, np.nan), "n": n,
                                   "method": "average-derivative (insufficient data)"})
    h = float(bandwidth) if bandwidth is not None else _silverman(X[:, 0])
    if h <= 0:
        h = max(_silverman(X[:, 0]), 1e-6)
    # Pairwise differences and Gaussian kernel
    diffs = X[:, None, :] - X[None, :, :]            # (n, n, p)
    sq = (diffs * diffs).sum(axis=2) / (h * h)
    K = np.exp(-0.5 * sq) / ((2 * np.pi) ** (p / 2) * h ** p)
    # grad f_hat at X_i wrt x = -(1/(n h^2)) sum_j (X_i - X_j) K_h(X_i-X_j)
    np.fill_diagonal(K, 0.0)
    grad_f = -(1.0 / (n * h * h)) * np.einsum('ijk,ij->ik', diffs, K)
    # Influence-function form:  delta_hat = -(2/n) sum_i Y_i * grad f_hat(X_i)
    delta = -(2.0 / n) * (y[:, None] * grad_f).sum(axis=0)
    # IID-style SE
    psi = -2 * y[:, None] * grad_f                    # (n, p)
    cov = np.cov(psi, rowvar=False) / n
    if p == 1:
        se = np.array([float(np.sqrt(max(cov, 0)))])
    else:
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    return RichResult(payload={
        "estimate": delta.astype(float) if delta.size > 1 else float(delta[0]),
        "se": se.astype(float) if se.size > 1 else float(se[0]),
        "n": n, "bandwidth": h,
        "method": "Powell-Stock-Stoker density-weighted average derivative",
    })


def cheatsheet():
    return "hrzi2: average-derivative estimator"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(5)
    n = 600
    X = rng.standard_normal((n, 2))
    beta = np.array([1.0, -0.5])
    y = X @ beta + 0.1 * rng.standard_normal(n)
    res = horowitz_average_derivative(X, y)
    print(res)
    # density-weighted AD for linear DGP ≈ beta * E[f(X)] (constant scale,
    # ratio of components preserved)
    ratio = res["estimate"][0] / res["estimate"][1]
    target = beta[0] / beta[1]
    assert abs(ratio - target) < 1.0, f"ratio={ratio}"
