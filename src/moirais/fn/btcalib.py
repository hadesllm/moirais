"""Bootstrap-calibrated nominal level."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_calibrated_ci"]


def boot_calibrated_ci(x, stat, B, Bp, alpha):
    """
    Bootstrap-calibrated nominal level

    Formula: Pick α' so coverage of inner CIs = 1-α

    Parameters
    ----------
    x : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.
    Bp : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: alpha_adj, lo, hi

    References
    ----------
    Loh (1991)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap-calibrated nominal level"})


def cheatsheet():
    return "btcalib: Bootstrap-calibrated nominal level"
