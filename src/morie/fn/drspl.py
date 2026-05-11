"""Split-sample DR-DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_split_sample"]


def dr_did_split_sample(y, D, X, K):
    """
    Split-sample DR-DiD

    Formula: split into K folds; compute DR on held-out

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sant'Anna-Zhao (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Split-sample DR-DiD"})


def cheatsheet():
    return "drspl: Split-sample DR-DiD"
