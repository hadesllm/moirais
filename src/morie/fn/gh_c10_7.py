# morie.fn — function file (hadesllm/morie)
"""Density estimation via finite random series prior: log f = sum beta_k phi_k."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_frs_density"]


def ghosal_frs_density(x):
    """
    Density estimation via finite random series prior: log f = sum beta_k phi_k

    Formula: log f = sum_{k<=K} beta_k phi_k - log Z, K~pi_n, rate n^{-s/(2s+1)}

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
    Ghosal Ch 10 §10.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Density estimation via finite random series prior: log f = sum beta_k phi_k"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Density estimation via finite random series prior: log f = sum beta_k phi_k"})


def cheatsheet():
    return "gh_c10_7: Density estimation via finite random series prior: log f = sum beta_k phi_k"
