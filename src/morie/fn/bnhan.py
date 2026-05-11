# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Binary's maximum rank correlation estimator."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def bnhan(
    y: np.ndarray,
    X: np.ndarray,
) -> dict:
    r"""
's maximum rank correlation (MRC) estimator for binary response.

    Maximises the rank correlation between :math:`Y` and :math:`X'\beta`:

    .. math::

        R_n(\beta) = \binom{n}{2}^{-1}
        \sum_{i<j} \mathbf{1}[(Y_i - Y_j)(X_i'\beta - X_j'\beta) > 0]

    Rank-based, so robust to monotone transformations of the link.

    Parameters
    ----------
    y : np.ndarray
        Binary (or ordinal) response (n,).
    X : np.ndarray
        Covariates (n, p), p >= 2.

    Returns
    -------
    dict
        ``beta`` (normalised), ``rank_correlation``, ``n_obs``.

    References
    ----------
, A. K. (1987). Non-parametric analysis of a generalized
        regression model. JoE, 35, 303-316.
    Horowitz (2009). Ch 4.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X must have same n.")
    if p < 2:
        raise ValueError("Need p >= 2 covariates.")
    if n < 5:
        raise ValueError("Need at least 5 observations.")

    y_diff = y[:, None] - y[None, :]

    def neg_rc(b):
        b_norm = b / (np.linalg.norm(b) + 1e-15)
        idx = X @ b_norm
        idx_diff = idx[:, None] - idx[None, :]
        concordant = (y_diff * idx_diff > 0).astype(float)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return -float(concordant[mask].mean())

    b0 = np.zeros(p)
    b0[0] = 1.0
    res = minimize(neg_rc, b0, method="L-BFGS-B",
                   options={"maxiter": 200})
    beta = res.x / (np.linalg.norm(res.x) + 1e-15)
    rc = -float(res.fun)

    return {
        "beta": beta.tolist(),
        "rank_correlation": rc,
        "n_obs": n,
    }


bnhan_fn = bnhan


def cheatsheet() -> str:
    return "bnhan({y, X}) ->'s maximum rank correlation (binary)."
