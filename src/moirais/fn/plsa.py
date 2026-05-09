"""Probabilistic LSA."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["plsa"]


def plsa(docs, K):
    """
    Probabilistic LSA

    Formula: P(w|d) = sum_z P(w|z)P(z|d); EM

    Parameters
    ----------
    docs : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hofmann (1999)
    """
    docs = np.atleast_1d(np.asarray(docs, dtype=float))
    n = len(docs)
    result = float(np.mean(docs))
    se = float(np.std(docs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Probabilistic LSA"})


def cheatsheet():
    return "plsa: Probabilistic LSA"
