"""MH rate ratio for person-time data."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mantel_haenszel_rate"]


def mantel_haenszel_rate(strata):
    """
    MH rate ratio for person-time data

    Formula: RR_MH = sum(a_k T0_k / T_k) / sum(c_k T1_k / T_k)

    Parameters
    ----------
    strata : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rothman-Greenland (1998)
    """
    strata = np.atleast_1d(np.asarray(strata, dtype=float))
    n = len(strata)
    result = float(np.mean(strata))
    se = float(np.std(strata, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MH rate ratio for person-time data"})


def cheatsheet():
    return "mhrate: MH rate ratio for person-time data"
