# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Binary maximum score estimator (Manski 1975)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution


def bnmax(
    y: np.ndarray,
    X: np.ndarray,
    *,
    seed: int | None = None,
) -> dict:
    r"""
    Maximum score estimator for binary response (Manski 1975/1985).

    Maximises:

    .. math::

        S_n(\beta) = \frac{1}{n} \sum_{i=1}^n
        \left[(2Y_i - 1) \cdot \mathbf{1}(X_i'\beta \geq 0)\right]

    The estimator is cube-root-n consistent but not asymptotically
    normal. Use the smoothed version (``bnsmo``) for inference.

    Parameters
    ----------
    y : np.ndarray
        Binary response (n,), values in {0, 1}.
    X : np.ndarray
        Covariates (n, p).
    seed : int or None
        RNG seed for differential evolution.

    Returns
    -------
    dict
        ``beta`` (normalised), ``score`` (maximum score value),
        ``n_obs``.

    References
    ----------
    Manski, C. F. (1975). Maximum score estimation of the stochastic
        utility model of choice. JoE, 3, 205-228.
    Horowitz (2009). Ch 4, eq. 4.11-4.14.
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
    if n < 5:
        raise ValueError("Need at least 5 observations.")

    signs = 2 * y - 1

    def neg_score(b):
        b_norm = b / (np.linalg.norm(b) + 1e-15)
        idx = X @ b_norm
        return -float(np.mean(signs * (idx >= 0).astype(float)))

    bounds = [(-1, 1)] * p
    res = differential_evolution(neg_score, bounds, seed=seed,
                                 maxiter=200, tol=1e-8, polish=True)
    beta = res.x / (np.linalg.norm(res.x) + 1e-15)
    score = -float(res.fun)

    return {
        "beta": beta.tolist(),
        "score": score,
        "n_obs": n,
    }


bnmax_fn = bnmax


def cheatsheet() -> str:
    return "bnmax({y, X}) -> Maximum score binary estimator (Manski)."
