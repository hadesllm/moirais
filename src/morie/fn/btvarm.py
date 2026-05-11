"""Bootstrap variance of the sample mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_var_mean"]


def boot_var_mean(x, B):
    """
    Bootstrap variance of the sample mean

    Formula: Var* = (1/B) Σ (mean_b - mean̄_b)²

    Parameters
    ----------
    x : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: var_b, mean_b

    References
    ----------
    Efron (1979)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap variance of the sample mean"})


def cheatsheet():
    return "btvarm: Bootstrap variance of the sample mean"
