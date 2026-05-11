# morie.fn — function file (hadesllm/morie)
"""Z-estimator asymptotic distribution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_z_estimator"]


def kosorok_z_estimator(x, y):
    """
    Z-estimator asymptotic distribution

    Formula: sqrt(n)(theta_n - theta_0) -> N(0, V)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Z-estimator asymptotic distribution"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Z-estimator asymptotic distribution"})


def cheatsheet():
    return "ksr09: Z-estimator asymptotic distribution"
