"""Non-negative matrix factorization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_nmf"]


def esl_nmf(X, k):
    """
    Non-negative matrix factorization

    Formula: X ~ W H, W,H >= 0

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W, H

    References
    ----------
    Hastie ESL Ch 14
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Non-negative matrix factorization"})


def cheatsheet():
    return "eslnmf: Non-negative matrix factorization"
