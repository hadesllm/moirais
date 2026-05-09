"""Prevalence ratio."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prevalence_ratio"]


def prevalence_ratio(prev_exposed, prev_unexposed):
    """
    Prevalence ratio

    Formula: PR = prev_e / prev_u

    Parameters
    ----------
    prev_exposed : array-like
        Input data.
    prev_unexposed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Barros-Hirakata (2003)
    """
    prev_exposed = np.atleast_1d(np.asarray(prev_exposed, dtype=float))
    n = len(prev_exposed)
    result = float(np.mean(prev_exposed))
    se = float(np.std(prev_exposed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prevalence ratio"})


def cheatsheet():
    return "prratio: Prevalence ratio"
