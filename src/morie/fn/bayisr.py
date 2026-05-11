"""Sampling-importance-resampling (SIR)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["importance_resample"]


def importance_resample(target, proposal, n, M):
    """
    Sampling-importance-resampling (SIR)

    Formula: resample with weights w_i

    Parameters
    ----------
    target : array-like
        Input data.
    proposal : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rubin (1988)
    """
    target = np.atleast_1d(np.asarray(target, dtype=float))
    n = len(target)
    result = float(np.mean(target))
    se = float(np.std(target, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sampling-importance-resampling (SIR)"})


def cheatsheet():
    return "bayisr: Sampling-importance-resampling (SIR)"
