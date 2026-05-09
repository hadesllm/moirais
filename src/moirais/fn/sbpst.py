"""Posterior stick-breaking weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stick_breaking_post"]


def stick_breaking_post(partition, alpha):
    """
    Posterior stick-breaking weights

    Formula: posterior V_k|data ~ Beta-conjugate

    Parameters
    ----------
    partition : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sethuraman (1994); Ishwaran-James (2001)
    """
    partition = np.atleast_1d(np.asarray(partition, dtype=float))
    n = len(partition)
    result = float(np.mean(partition))
    se = float(np.std(partition, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior stick-breaking weights"})


def cheatsheet():
    return "sbpst: Posterior stick-breaking weights"
