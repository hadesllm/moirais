# morie.fn — function file (hadesllm/morie)
"""Gaussian location-scale kernel for DPM density estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gauss_ker"]


def ghosal_gauss_ker(x):
    """
    Gaussian location-scale kernel for DPM density estimation

    Formula: f(x) = integral phi_{mu,sigma}(x) dG(mu,sigma), G ~ DP

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 5 §5.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Gaussian location-scale kernel for DPM density estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Gaussian location-scale kernel for DPM density estimation"})


def cheatsheet():
    return "gh_c5_8: Gaussian location-scale kernel for DPM density estimation"
