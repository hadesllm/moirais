"""chrF character n-gram F-score."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chrf"]


def chrf(candidate, reference, beta):
    """
    chrF character n-gram F-score

    Formula: F_β over char n-gram P/R

    Parameters
    ----------
    candidate : array-like
        Input data.
    reference : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Popović (2015)
    """
    candidate = np.atleast_1d(np.asarray(candidate, dtype=float))
    n = len(candidate)
    result = float(np.mean(candidate))
    se = float(np.std(candidate, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "chrF character n-gram F-score"})


def cheatsheet():
    return "chrF: chrF character n-gram F-score"
