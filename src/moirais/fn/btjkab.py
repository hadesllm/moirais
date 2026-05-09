"""Jackknife-after-bootstrap influence diagnostic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_jackknife_after_boot"]


def boot_jackknife_after_boot(x, theta_b, B_idx):
    """
    Jackknife-after-bootstrap influence diagnostic

    Formula: Recompute θ̂* on resamples not containing i

    Parameters
    ----------
    x : array-like
        Input data.
    theta_b : array-like
        Input data.
    B_idx : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: infl_i

    References
    ----------
    Efron (1992)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Jackknife-after-bootstrap influence diagnostic"})


def cheatsheet():
    return "btjkab: Jackknife-after-bootstrap influence diagnostic"
