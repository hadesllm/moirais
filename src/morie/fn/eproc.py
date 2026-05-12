# morie.fn -- function file (hadesllm/morie)
"""Empirical process: sqrt(n) * (Fn - F) at evaluation points."""

from __future__ import annotations

import numpy as np

__all__ = ["eproc"]


def eproc(x: np.ndarray, eval_points: np.ndarray | None = None, cdf=None, *, cdf_func: callable | None = None) -> dict:
    r"""
    Compute the empirical process :math:`\mathbb{G}_n(t) = \sqrt{n}(\hat{F}_n(t) - F(t))`.

    When ``cdf_func`` is None, :math:`F` defaults to the standard normal CDF.

    :param x: 1-D array of observations.
    :param eval_points: Points at which to evaluate. Defaults to sorted unique x.
    :param cdf_func: True CDF function F(t). Default ``scipy.stats.norm.cdf``.
    :return: Dict with keys ``points``, ``process``, ``ecdf``, ``true_cdf``, ``n``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 5. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    if cdf_func is None:
        from scipy.stats import norm
        cdf_func = norm.cdf

    n = x.size
    x_sorted = np.sort(x)

    if eval_points is None:
        eval_points = np.unique(x_sorted)
    else:
        eval_points = np.asarray(eval_points, dtype=float).ravel()

    ecdf_vals = np.searchsorted(x_sorted, eval_points, side="right") / n
    true_cdf_vals = cdf_func(eval_points)
    process = np.sqrt(n) * (ecdf_vals - true_cdf_vals)

    return {
        "points": eval_points,
        "process": process,
        "ecdf": ecdf_vals,
        "true_cdf": true_cdf_vals,
        "n": n,
    }


def cheatsheet() -> str:
    return "eproc({x}) -> Empirical process sqrt(n)*(Fn - F)."
