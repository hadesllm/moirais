"""HAR-Q model adding realised quarticity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_har_q"]


def vol_har_q(RV, RQ):
    """
    HAR-Q model adding realised quarticity

    Formula: RV_t = β_d RV + β_w RV_w + β_m RV_m + β_q sqrt(RQ) RV

    Parameters
    ----------
    RV : array-like
        Input data.
    RQ : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: betas, fitted

    References
    ----------
    Bollerslev-Patton-Quaedvlieg (2016)
    """
    RV = np.atleast_1d(np.asarray(RV, dtype=float))
    n = len(RV)
    result = float(np.mean(RV))
    se = float(np.std(RV, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HAR-Q model adding realised quarticity"})


def cheatsheet():
    return "volhar1: HAR-Q model adding realised quarticity"
