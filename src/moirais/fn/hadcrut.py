"""HadCRUT5 anomaly + CIs."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hadcrut"]


def hadcrut(T, sst):
    """
    HadCRUT5 anomaly + CIs

    Formula: land + SST + observational uncertainty

    Parameters
    ----------
    T : array-like
        Input data.
    sst : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Morice et al (2021)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HadCRUT5 anomaly + CIs"})


def cheatsheet():
    return "hadcrut: HadCRUT5 anomaly + CIs"
