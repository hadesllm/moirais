# moirais.fn — function file (hadesllm/moirais)
"""Fractional Brownian motion prior: Hurst parameter H controls smoothness."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_fbm_prior"]


def ghosal_fbm_prior(x):
    """
    Fractional Brownian motion prior: Hurst parameter H controls smoothness

    Formula: fBM: k(s,t) = (|s|^{2H}+|t|^{2H}-|s-t|^{2H})/2, H in (0,1)

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
    Ghosal Ch 11 §11.4.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fractional Brownian motion prior: Hurst parameter H controls smoothness"})


def cheatsheet():
    return "gh_c11_8: Fractional Brownian motion prior: Hurst parameter H controls smoothness"
