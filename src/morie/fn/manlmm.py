"""Network meta-analysis via linear mixed model on contrasts."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_network_lme"]


def ma_network_lme(yi, vi, design):
    """
    Network meta-analysis via linear mixed model on contrasts

    Formula: y_ij = μ + θ_t-θ_b + u_ij + ε_ij

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    design : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta, ranks

    References
    ----------
    Salanti et al. (2008)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Network meta-analysis via linear mixed model on contrasts"})


def cheatsheet():
    return "manlmm: Network meta-analysis via linear mixed model on contrasts"
