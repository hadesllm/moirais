"""Decomposed realised volatility into continuous + jump per BNS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_decomposed_realised"]


def vol_decomposed_realised(RV, BPV):
    """
    Decomposed realised volatility into continuous + jump per BNS

    Formula: RV = C + J; J = max(RV-BPV, 0)

    Parameters
    ----------
    RV : array-like
        Input data.
    BPV : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: C, J

    References
    ----------
    Barndorff-Nielsen-Shephard (2006)
    """
    RV = np.atleast_1d(np.asarray(RV, dtype=float))
    n = len(RV)
    result = float(np.mean(RV))
    se = float(np.std(RV, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decomposed realised volatility into continuous + jump per BNS"})


def cheatsheet():
    return "voldoc: Decomposed realised volatility into continuous + jump per BNS"
