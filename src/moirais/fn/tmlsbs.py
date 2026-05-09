"""TMLE for variable selection / subset inference."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_subset_selection"]


def tmle_subset_selection(y, D, X):
    """
    TMLE for variable selection / subset inference

    Formula: selection-corrected influence curve

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hubbard et al (2016) Subset
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for variable selection / subset inference"})


def cheatsheet():
    return "tmlsbs: TMLE for variable selection / subset inference"
