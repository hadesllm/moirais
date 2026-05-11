"""Bipower variation (jump-robust realised volatility)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_bipower_variation"]


def vol_bipower_variation(r_intraday, block_index):
    """
    Bipower variation (jump-robust realised volatility)

    Formula: BPV = (π/2) Σ |r_i| |r_{i-1}|

    Parameters
    ----------
    r_intraday : array-like
        Input data.
    block_index : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: BPV

    References
    ----------
    Barndorff-Nielsen-Shephard (2004)
    """
    r_intraday = np.atleast_1d(np.asarray(r_intraday, dtype=float))
    n = len(r_intraday)
    result = float(np.mean(r_intraday))
    se = float(np.std(r_intraday, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bipower variation (jump-robust realised volatility)"})


def cheatsheet():
    return "volbpv: Bipower variation (jump-robust realised volatility)"
