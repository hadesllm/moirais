# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bandwidth-selected nonparametric regression."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["bwreg"]


def bwreg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    eval_points: np.ndarray | None = None,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    n_eval: int = 100,
) -> dict[str, Any]:
    r"""
    Nadaraya-Watson kernel regression with bandwidth selection.

    .. math::

        \hat{m}(x) = \frac{\sum_{i=1}^n K_h(x - X_i) Y_i}
        {\sum_{i=1}^n K_h(x - X_i)}

    If bandwidth is None, uses Silverman's rule of thumb.

    :param x: Predictor values, shape (n,).
    :param y: Response values, shape (n,).
    :param eval_points: Points at which to evaluate. Default: linspace over x range.
    :param bandwidth: Bandwidth h. If None, auto-selected.
    :param kernel: ``"gaussian"`` (default) or ``"epanechnikov"``.
    :param n_eval: Number of evaluation points. Default 100.
    :return: Dict with ``eval_points``, ``fitted``, ``bandwidth``, ``n``.
    :raises ValueError: If arrays are empty or mismatched.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 13. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if x.size != y.size:
        raise ValueError("x and y must have same length.")

    n = x.size

    if bandwidth is None:
        bandwidth = 1.06 * np.std(x, ddof=1) * n ** (-1.0 / 5.0)

    if eval_points is None:
        eval_points = np.linspace(np.min(x), np.max(x), n_eval)
    else:
        eval_points = np.asarray(eval_points, dtype=float).ravel()

    if kernel == "gaussian":
        def K(u):
            return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    elif kernel == "epanechnikov":
        def K(u):
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
    else:
        raise ValueError(f"kernel must be 'gaussian' or 'epanechnikov', got '{kernel}'.")

    fitted = np.zeros(len(eval_points))
    for k, xp in enumerate(eval_points):
        w = K((xp - x) / bandwidth)
        denom = np.sum(w)
        if denom > 0:
            fitted[k] = np.sum(w * y) / denom
        else:
            fitted[k] = np.nan

    return {
        "eval_points": eval_points,
        "fitted": fitted,
        "bandwidth": float(bandwidth),
        "n": n,
    }


def cheatsheet() -> str:
    return "bwreg({x, y}) -> Bandwidth-selected kernel regression."
