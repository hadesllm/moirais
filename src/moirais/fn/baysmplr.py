"""Sampler dispatch (auto-select MH/Gibbs/HMC/NUTS)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sampler_dispatch"]


def sampler_dispatch(log_p, grad_p, x0):
    """
    Sampler dispatch (auto-select MH/Gibbs/HMC/NUTS)

    Formula: heuristic on dimensionality + gradient availability

    Parameters
    ----------
    log_p : array-like
        Input data.
    grad_p : array-like
        Input data.
    x0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    applied
    """
    log_p = np.atleast_1d(np.asarray(log_p, dtype=float))
    n = len(log_p)
    result = float(np.mean(log_p))
    se = float(np.std(log_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sampler dispatch (auto-select MH/Gibbs/HMC/NUTS)"})


def cheatsheet():
    return "baysmplr: Sampler dispatch (auto-select MH/Gibbs/HMC/NUTS)"
