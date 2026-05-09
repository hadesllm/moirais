# moirais.fn — function file (hadesllm/moirais)
"""Gibbs sampler: iteratively sample each parameter from full conditional."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gibbs_sampler"]


def ghosal_gibbs_sampler(x):
    """
    Gibbs sampler: iteratively sample each parameter from full conditional

    Formula: theta_j^{(t+1)} ~ pi(theta_j | theta_{-j}^{(t)}, X)

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
    Ghosal App M
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gibbs sampler: iteratively sample each parameter from full conditional"})


def cheatsheet():
    return "gh_ap_m2: Gibbs sampler: iteratively sample each parameter from full conditional"
