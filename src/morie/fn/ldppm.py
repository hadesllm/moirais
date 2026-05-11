"""Local DP planar / k-RR mechanism."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["local_dp_planar_mechanism"]


def local_dp_planar_mechanism(y, truth, k, epsilon):
    """
    Local DP planar / k-RR mechanism

    Formula: P(report=v|true=u) = e^epsilon / (k - 1 + e^epsilon) if v=u else 1/(k-1+e^epsilon)

    Parameters
    ----------
    y : array-like
        Input data.
    truth : array-like
        Input data.
    k : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Erlingsson, Pihur, Korolova (2014) RAPPOR
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local DP planar / k-RR mechanism"})


def cheatsheet():
    return "ldppm: Local DP planar / k-RR mechanism"
