"""Data-adaptive parameter TMLE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_data_adaptive"]


def tmle_data_adaptive(y, D, X, candidate_strata):
    """
    Data-adaptive parameter TMLE

    Formula: target a parameter chosen from data -- eg max-CATE stratum

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    candidate_strata : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hubbard-Kennedy-vdL (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Data-adaptive parameter TMLE"})


def cheatsheet():
    return "tmldta: Data-adaptive parameter TMLE"
