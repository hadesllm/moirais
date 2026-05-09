"""Tsallis q-entropy."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tsallis_entropy"]


def tsallis_entropy(p, q):
    """
    Tsallis q-entropy

    Formula: S_q = (1 - sum p^q)/(q-1)

    Parameters
    ----------
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tsallis (1988)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tsallis q-entropy"})


def cheatsheet():
    return "tsalls: Tsallis q-entropy"
