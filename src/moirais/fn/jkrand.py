"""Jackknife replicate weights variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["jackknife_repl"]


def jackknife_repl(theta_replicates):
    """
    Jackknife replicate weights variance

    Formula: Var = (n-1)/n sum (theta_i - thetabar)^2

    Parameters
    ----------
    theta_replicates : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wolter (2007)
    """
    theta_replicates = np.atleast_1d(np.asarray(theta_replicates, dtype=float))
    n = len(theta_replicates)
    result = float(np.mean(theta_replicates))
    se = float(np.std(theta_replicates, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Jackknife replicate weights variance"})


def cheatsheet():
    return "jkrand: Jackknife replicate weights variance"
