# moirais.fn — function file (hadesllm/moirais)
"""Latent growth curve model (simplified OLS-based)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def latent_growth(Y_longitudinal: np.ndarray) -> DescriptiveResult:
    """Simplified latent growth model via OLS per subject.

    Fits intercept + slope per subject over equally spaced waves.

    Parameters
    ----------
    Y_longitudinal : (n_subjects, n_waves) array

    Returns
    -------
    DescriptiveResult
    """
    Y = np.asarray(Y_longitudinal, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Need (n_subjects, n_waves) matrix.")
    n, T = Y.shape
    if T < 2:
        raise ValueError("Need >= 2 time waves.")

    time = np.arange(T, dtype=float)
    X = np.column_stack([np.ones(T), time])

    intercepts = np.empty(n)
    slopes = np.empty(n)
    for i in range(n):
        beta = np.linalg.lstsq(X, Y[i], rcond=None)[0]
        intercepts[i] = beta[0]
        slopes[i] = beta[1]

    return DescriptiveResult(
        name="lcgm",
        value=float(np.mean(slopes)),
        extra={
            "mean_intercept": float(np.mean(intercepts)),
            "mean_slope": float(np.mean(slopes)),
            "var_intercept": float(np.var(intercepts, ddof=1)),
            "var_slope": float(np.var(slopes, ddof=1)),
            "cov_int_slope": float(np.cov(intercepts, slopes)[0, 1]),
            "n": n,
            "n_waves": T,
        },
    )


lcgm = latent_growth


def cheatsheet() -> str:
    return "latent_growth({}) -> Latent growth curve model (simplified OLS-based)."
