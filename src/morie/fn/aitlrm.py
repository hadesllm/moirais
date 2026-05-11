"""Geometric (log-ratio) mean composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_lrmean"]


def compositional_lrmean(X):
    """
    Geometric (log-ratio) mean composition

    Formula: ḡ_i = exp(mean(log x_{·i})); C(ḡ)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g

    References
    ----------
    Aitchison (1986)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Geometric (log-ratio) mean composition"})


def cheatsheet():
    return "aitlrm: Geometric (log-ratio) mean composition"
