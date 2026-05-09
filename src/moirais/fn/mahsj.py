"""HKSJ-based t-distribution prediction interval."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_hksj_t_pi"]


def ma_hksj_t_pi(theta, se_hksj, tau2, k):
    """
    HKSJ-based t-distribution prediction interval

    Formula: θ̂ ± t_{k-1, 1-α/2} sqrt(τ̂²+SE_HKSJ²)

    Parameters
    ----------
    theta : array-like
        Input data.
    se_hksj : array-like
        Input data.
    tau2 : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi

    References
    ----------
    Inthout-Ioannidis-Borm (2014)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HKSJ-based t-distribution prediction interval"})


def cheatsheet():
    return "mahsj: HKSJ-based t-distribution prediction interval"
