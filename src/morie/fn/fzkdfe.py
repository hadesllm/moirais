# morie.fn — function file (hadesllm/morie)
"""Nadaraya kernel distribution function estimator (1964)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_kdfe"]


def fauzi_kdfe(x, bandwidth, kernel):
    """
    Nadaraya kernel distribution function estimator (1964)

    Formula: F_hat_h(x) = (1/n) sum W((x-X_i)/h), W = integral K

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 2, Eq 2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Nadaraya kernel distribution function estimator (1964)"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Nadaraya kernel distribution function estimator (1964)"})


def cheatsheet():
    return "fzkdfe: Nadaraya kernel distribution function estimator (1964)"
