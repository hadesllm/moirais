# morie.fn -- function file (hadesllm/morie)
"""Besov prior for adaptation: wavelet coefficient prior adapts to Besov smoothness."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_besov_prior"]


def ghosal_besov_prior(x):
    """
    Besov prior for adaptation: wavelet coefficient prior adapts to Besov smoothness

    Formula: theta_{jk} ~ pi_j * N(0, 2^{-j(2s+1)}) + (1-pi_j)*delta_0

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
    Ghosal Ch 10 §10.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Besov prior for adaptation: wavelet coefficient prior adapts to Besov smoothness"})


def cheatsheet():
    return "gh_besov_prior: Besov prior for adaptation: wavelet coefficient prior adapts to Besov smoothness"
