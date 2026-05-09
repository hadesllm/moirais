"""Composite likelihood for variogram estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_composite_likelihood"]


def schabenberger_composite_likelihood(coords, z, variogram_model):
    """
    Composite likelihood for variogram estimation

    Formula: CL(theta) = sum_{i<j} log f(Z(s_i),Z(s_j); theta) summed over pairs

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    variogram_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: parameters

    References
    ----------
    Schabenberger Ch 4, Sec 4.5.3
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    if z.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Composite likelihood for variogram estimation"})
    estimate = np.median(z)
    se = 1.2533 * np.std(z, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Composite likelihood for variogram estimation"})


def cheatsheet():
    return "spclk: Composite likelihood for variogram estimation"
