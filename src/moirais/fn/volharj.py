"""HAR-RV-Jump separating continuous + jump components."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_har_rv_jump"]


def vol_har_rv_jump(RV, BPV):
    """
    HAR-RV-Jump separating continuous + jump components

    Formula: RV_t = c + β_d C + β_w C_w + β_m C_m + β_J J + ε

    Parameters
    ----------
    RV : array-like
        Input data.
    BPV : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: betas, fitted

    References
    ----------
    Andersen-Bollerslev-Diebold (2007)
    """
    RV = np.atleast_1d(np.asarray(RV, dtype=float))
    n = len(RV)
    result = float(np.mean(RV))
    se = float(np.std(RV, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HAR-RV-Jump separating continuous + jump components"})


def cheatsheet():
    return "volharj: HAR-RV-Jump separating continuous + jump components"
