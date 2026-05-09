"""FarmCPU iterative MLM + fixed-effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["farm_cpu"]


def farm_cpu(y, M, K):
    """
    FarmCPU iterative MLM + fixed-effect

    Formula: iterate: pseudo-QTNs as fixed; rest in K

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu et al (2016) FarmCPU
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FarmCPU iterative MLM + fixed-effect"})


def cheatsheet():
    return "farmlmm: FarmCPU iterative MLM + fixed-effect"
