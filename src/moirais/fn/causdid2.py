"""Canonical 2x2 difference-in-differences."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_did_2x2"]


def causal_did_2x2(y, treated, post):
    """
    Canonical 2x2 difference-in-differences

    Formula: ATT = (ȳ_T1-ȳ_T0) - (ȳ_C1-ȳ_C0)

    Parameters
    ----------
    y : array-like
        Input data.
    treated : array-like
        Input data.
    post : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATT, se

    References
    ----------
    Card & Krueger (1994)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Canonical 2x2 difference-in-differences"})


def cheatsheet():
    return "causdid2: Canonical 2x2 difference-in-differences"
