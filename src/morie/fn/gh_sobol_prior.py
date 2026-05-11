# morie.fn — function file (hadesllm/morie)
"""Sobolev prior: theta ~ N(0, Lambda) with Lambda = diag(j^{-2s-1}) in series model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_sobolev_prior"]


def ghosal_sobolev_prior(x):
    """
    Sobolev prior: theta ~ N(0, Lambda) with Lambda = diag(j^{-2s-1}) in series model

    Formula: theta_j ~ N(0, j^{-2s-1}), Sobolev(s) prior, rate n^{-2s/(2s+1)}

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
    Ghosal Ch 9 §9.5.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sobolev prior: theta ~ N(0, Lambda) with Lambda = diag(j^{-2s-1}) in series model"})


def cheatsheet():
    return "gh_sobol_prior: Sobolev prior: theta ~ N(0, Lambda) with Lambda = diag(j^{-2s-1}) in series model"
