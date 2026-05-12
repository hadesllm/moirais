r"""Smoothed maximum score estimator (Horowitz 1992).

Replaces the discontinuous sign function in Manski's maximum score
with a smooth kernel approximation:

.. math::

    S_n^s(\\beta) = n^{-1} \\sum_{i=1}^n
        \\bigl[Y_i - \\tfrac{1}{2}\\bigr] \\,
        K\\!\\left(\\frac{X_i^\\top \\beta}{h_n}\\right)

where :math:`K` is a CDF kernel (standard normal) and :math:`h_n`
is a bandwidth that shrinks with :math:`n`.

References
----------
Horowitz, J. L. (1992). A smoothed maximum score estimator for the
    binary response model. *Econometrica*, 60(3), 505--531.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats


def smscr(Y: np.ndarray, X: np.ndarray, cdf=None, *, bandwidth: float | None = None, random_state: int = 42) -> dict[str, Any]:
    r"""Smoothed maximum score estimator for binary response.

    Parameters
    ----------
    Y : np.ndarray
        Binary outcome (0/1), shape ``(n,)``.
    X : np.ndarray
        Covariate matrix, shape ``(n, p)``.
    bandwidth : float or None
        Smoothing bandwidth.  If *None*, uses :math:`n^{-1/5}`.
    random_state : int
        Random seed for initial values.

    Returns
    -------
    dict[str, Any]
        ``beta`` (unit-norm), ``score``, ``bandwidth``, ``n``, ``p``,
        ``method``.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if len(Y) != n:
        raise ValueError(f"Y length {len(Y)} != X rows {n}.")

    if bandwidth is None:
        bandwidth = float(n ** (-1.0 / 5.0))

    centered = Y - 0.5

    def neg_score(b_raw: np.ndarray) -> float:
        b = b_raw / (np.linalg.norm(b_raw) + 1e-12)
        idx = X @ b / bandwidth
        return -float(np.mean(centered * stats.norm.cdf(idx)))

    rng = np.random.default_rng(random_state)
    best_res = None
    for _ in range(20):
        b0 = rng.standard_normal(p)
        b0 /= np.linalg.norm(b0) + 1e-12
        res = optimize.minimize(neg_score, b0, method="Nelder-Mead",
                                options={"maxiter": 500, "xatol": 1e-6})
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    beta = best_res.x / (np.linalg.norm(best_res.x) + 1e-12)
    if beta[0] < 0:
        beta = -beta

    return {
        "beta": beta,
        "score": float(-best_res.fun),
        "bandwidth": bandwidth,
        "n": n,
        "p": p,
        "method": "SmoothedMaximumScore",
    }


smscr_fn = smscr


def cheatsheet() -> str:
    return "smscr(Y, X) -> Smoothed maximum score estimator (Horowitz 1992)."
