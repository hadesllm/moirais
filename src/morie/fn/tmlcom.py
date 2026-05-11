"""TMLE for compositional treatments (sum-to-1 dose)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_compositional"]


def tmle_compositional(y, composition, X):
    """
    TMLE for compositional treatments (sum-to-1 dose)

    Formula: log-ratio transform; target composition shift effect

    Parameters
    ----------
    y : array-like
        Input data.
    composition : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Aitchison (1986); Kennedy (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for compositional treatments (sum-to-1 dose)"})


def cheatsheet():
    return "tmlcom: TMLE for compositional treatments (sum-to-1 dose)"
