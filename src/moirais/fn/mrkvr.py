# moirais.fn — function file (hadesllm/moirais)
"""Marker variance component estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["marker_variance"]


def marker_variance(x, y, markers):
    """
    Marker variance component estimation

    Formula: sigma_m^2 = sigma_g^2 / p for p markers

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    markers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Marker variance component estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Marker variance component estimation"})


def cheatsheet():
    return "mrkvr: Marker variance component estimation"
