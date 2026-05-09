"""Spline regression (natural cubic)."""

from __future__ import annotations

import numpy as np


def splrg(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    n_knots: int | None = None,
    knots: np.ndarray | None = None,
    penalty: float = 0.0,
) -> dict:
    r"""
    Natural cubic spline regression.

    Fits a piecewise cubic polynomial that is linear beyond the boundary
    knots, with optional roughness penalty (smoothing spline when
    ``penalty > 0``).

    Uses a truncated power basis:

    .. math::

        f(x) = \beta_0 + \beta_1 x + \sum_{k=1}^{K}
        \gamma_k (x - \kappa_k)_+^3

    Parameters
    ----------
    x, y : np.ndarray
        Predictor and response (n,).
    x_eval : np.ndarray or None
        Evaluation points. Defaults to sorted x.
    n_knots : int or None
        Number of interior knots. Default min(n//4, 20).
    knots : np.ndarray or None
        Explicit knot locations. Overrides ``n_knots``.
    penalty : float
        Roughness penalty lambda >= 0.

    Returns
    -------
    dict
        ``x_eval``, ``y_hat``, ``coefficients``, ``knots``,
        ``penalty``, ``n_obs``.

    References
    ----------
    Green, P. & Silverman, B. W. (1994). Nonparametric Regression and
        Generalized Linear Models. Chapman & Hall.
    Horowitz (2009). Appendix.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length.")
    if n < 4:
        raise ValueError("Need at least 4 observations.")

    if knots is None:
        if n_knots is None:
            n_knots = min(n // 4, 20)
        n_knots = max(1, n_knots)
        probs = np.linspace(0, 100, n_knots + 2)[1:-1]
        knots = np.percentile(x, probs)
    else:
        knots = np.asarray(knots, dtype=float).ravel()

    def _tp(x_arr, k):
        return np.maximum(x_arr - k, 0.0) ** 3

    K = len(knots)
    X = np.column_stack([np.ones(n), x] + [_tp(x, k) for k in knots])

    if penalty > 0:
        P = np.zeros((2 + K, 2 + K))
        for i in range(K):
            for j in range(K):
                P[2 + i, 2 + j] = 36 * np.mean(
                    np.maximum(x - knots[i], 0) * np.maximum(x - knots[j], 0)
                )
        XtX = X.T @ X + penalty * P
    else:
        XtX = X.T @ X

    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]

    if x_eval is None:
        x_eval = np.sort(x)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    X_eval = np.column_stack([np.ones(len(x_eval)), x_eval] + [_tp(x_eval, k) for k in knots])
    y_hat = X_eval @ beta

    return {
        "x_eval": x_eval.tolist(),
        "y_hat": y_hat.tolist(),
        "coefficients": beta.tolist(),
        "knots": knots.tolist(),
        "penalty": float(penalty),
        "n_obs": n,
    }


splrg_fn = splrg


def cheatsheet() -> str:
    return "splrg({x, y}) -> Natural cubic spline regression."
