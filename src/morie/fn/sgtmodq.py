"""Newman-Girvan modularity score Q."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_modularity_q"]


def sgt_modularity_q(A, labels):
    """
    Newman-Girvan modularity score Q

    Formula: Q = (1/2m) Σ B_{ij} 1{c_i=c_j}

    Parameters
    ----------
    A : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q

    References
    ----------
    Newman-Girvan (2004)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Newman-Girvan modularity score Q"})


def cheatsheet():
    return "sgtmodq: Newman-Girvan modularity score Q"
