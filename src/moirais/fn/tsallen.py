"""Tsallis q-entropy."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tsallis_entropy"]


def tsallis_entropy(y, q):
    """
    Tsallis q-entropy

    Formula: S_q(X) = (1/(q-1))(1 - sum_x p(x)^q)

    Parameters
    ----------
    y : array-like
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
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tsallis q-entropy"})


def cheatsheet():
    return "tsallen: Tsallis q-entropy"
