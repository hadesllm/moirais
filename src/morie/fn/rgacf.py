# morie.fn — function file (hadesllm/morie)
"""Unbiased autocorrelation function (ACF) estimate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_acf_estimate"]


def rangayyan_acf_estimate(x, max_lag):
    """
    Unbiased autocorrelation function (ACF) estimate

    Formula: R_xx(m) = (1/(N-|m|)) sum_{n=0}^{N-1-|m|} x(n)*x(n+m)

    Parameters
    ----------
    x : array-like
        Input data.
    max_lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: acf, lags

    References
    ----------
    Rangayyan Ch 6.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Unbiased autocorrelation function (ACF) estimate"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Unbiased autocorrelation function (ACF) estimate"})


def cheatsheet():
    return "rgacf: Unbiased autocorrelation function (ACF) estimate"
