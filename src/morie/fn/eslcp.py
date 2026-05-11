"""Mallows Cp criterion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_mallows_cp"]


def esl_mallows_cp(RSS, d, n, sigma2):
    """
    Mallows Cp criterion

    Formula: C_p = (1/n)(RSS + 2 d sigma^2)

    Parameters
    ----------
    RSS : array-like
        Input data.
    d : array-like
        Input data.
    n : array-like
        Input data.
    sigma2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 7
    """
    RSS = np.atleast_1d(np.asarray(RSS, dtype=float))
    n = len(RSS)
    result = float(np.mean(RSS))
    se = float(np.std(RSS, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mallows Cp criterion"})


def cheatsheet():
    return "eslcp: Mallows Cp criterion"
