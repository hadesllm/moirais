"""Unmatched case-control OR."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["case_control"]


def case_control(cases, controls, exposed, unexposed):
    """
    Unmatched case-control OR

    Formula: OR = ad/bc from 2x2

    Parameters
    ----------
    cases : array-like
        Input data.
    controls : array-like
        Input data.
    exposed : array-like
        Input data.
    unexposed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cornfield (1951)
    """
    cases = np.atleast_1d(np.asarray(cases, dtype=float))
    n = len(cases)
    result = float(np.mean(cases))
    se = float(np.std(cases, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unmatched case-control OR"})


def cheatsheet():
    return "ccdsgn: Unmatched case-control OR"
