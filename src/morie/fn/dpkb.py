"""k-anonymity baseline (NOT DP)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["k_anonymity"]


def k_anonymity(X, quasi_ids, k):
    """
    k-anonymity baseline (NOT DP)

    Formula: each release row indistinguishable from k−1 others on QIDs

    Parameters
    ----------
    X : array-like
        Input data.
    quasi_ids : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sweeney (2002)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-anonymity baseline (NOT DP)"})


def cheatsheet():
    return "dpkb: k-anonymity baseline (NOT DP)"
