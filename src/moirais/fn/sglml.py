"""Lagrange multiplier tests for spatial dependence."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def lm_test_spatial(residuals: np.ndarray, W: np.ndarray, cdf=None) -> SpatialResult:
    r"""LM tests for spatial lag and error dependence.

    Parameters
    ----------
    residuals : np.ndarray
        OLS residuals, shape ``(n,)``.
    W : np.ndarray
        Row-standardized weights, shape ``(n, n)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is LM-error statistic.
        ``extra`` has ``lm_lag``, ``p_lag``, ``lm_error``, ``p_error``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

        "The cycle ends here." -- Kratos, God of War
    """
    from scipy.stats import chi2

    e = np.asarray(residuals, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64)
    n = len(e)
    sigma2 = np.sum(e**2) / n

    We = W @ e
    lm_err = float((e @ We) ** 2 / (sigma2**2 * np.trace(W.T @ W + W @ W)))
    lm_lag_num = (e @ We / sigma2) ** 2
    T = np.trace(W @ W + W.T @ W)
    lm_lag = float(lm_lag_num / T) if T > 0 else 0.0

    p_err = float(1.0 - chi2.cdf(lm_err, 1))
    p_lag = float(1.0 - chi2.cdf(lm_lag, 1))

    return SpatialResult(
        name="lm_test_spatial",
        statistic=lm_err,
        p_value=p_err,
        extra={
            "lm_lag": lm_lag,
            "p_lag": p_lag,
            "lm_error": lm_err,
            "p_error": p_err,
        },
    )


sglml = lm_test_spatial


def cheatsheet() -> str:
    return "lm_test_spatial({}) -> Lagrange multiplier tests for spatial dependence."
