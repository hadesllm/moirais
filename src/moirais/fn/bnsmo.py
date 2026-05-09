# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Binary smoothed maximum score estimator (Horowitz 1992)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def bnsmo(y: np.ndarray, X: np.ndarray, cdf=None, *, bandwidth: float | None = None, seed: int | None = None) -> dict:
    r"""
    Smoothed maximum score estimator (Horowitz 1992).

    Replaces the indicator in the maximum score objective with a
    smooth kernel function:

    .. math::

        \tilde{S}_n(\beta) = \frac{1}{n} \sum_{i=1}^n
        (2Y_i - 1) \, K\!\left(\frac{X_i'\beta}{h_n}\right)

    This estimator is asymptotically normal (unlike the unsmoothed
    maximum score), enabling standard inference.

    Parameters
    ----------
    y : np.ndarray
        Binary response (n,), values in {0, 1}.
    X : np.ndarray
        Covariates (n, p).
    bandwidth : float or None
        Smoothing bandwidth. If None, :math:`n^{-1/5}`.
    seed : int or None
        RNG seed.

    Returns
    -------
    dict
        ``beta`` (normalised), ``score``, ``bandwidth``, ``n_obs``.

    References
    ----------
    Horowitz, J. L. (1992). A smoothed maximum score estimator for the
        binary response model. Econometrica, 60, 505-531.
    Horowitz (2009). Ch 4, eq. 4.24-4.30.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X must have same n.")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0/1).")
    if n < 10:
        raise ValueError("Need at least 10 observations.")

    if bandwidth is None:
        bandwidth = float(n ** (-1 / 5))

    signs = 2 * y - 1

    def neg_score(b):
        b_norm = b / (np.linalg.norm(b) + 1e-15)
        idx = X @ b_norm
        k_vals = stats.norm.cdf(idx / bandwidth)
        return -float(np.mean(signs * k_vals))

    b0 = np.zeros(p)
    b0[0] = 1.0
    res = minimize(neg_score, b0, method="L-BFGS-B",
                   options={"maxiter": 200})
    beta = res.x / (np.linalg.norm(res.x) + 1e-15)
    score = -float(res.fun)

    return {
        "beta": beta.tolist(),
        "score": score,
        "bandwidth": float(bandwidth),
        "n_obs": n,
    }


bnsmo_fn = bnsmo


def cheatsheet() -> str:
    return "bnsmo({y, X}) -> Smoothed maximum score (Horowitz 1992)."
