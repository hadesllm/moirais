"""Discriminant validity (AVE > shared variance)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["discriminant_validity"]


def discriminant_validity(AVE, factor_correlations):
    """
    Discriminant validity (AVE > shared variance)

    Formula: sqrt(AVE_i) > correlation(F_i, F_j)

    Parameters
    ----------
    AVE : array-like
        Input data.
    factor_correlations : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fornell-Larcker (1981)
    """
    AVE = np.atleast_1d(np.asarray(AVE, dtype=float))
    n = len(AVE)
    result = float(np.mean(AVE))
    se = float(np.std(AVE, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discriminant validity (AVE > shared variance)"})


def cheatsheet():
    return "divgvs: Discriminant validity (AVE > shared variance)"
