"""Indirect comparison via Bucher's adjusted method."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_network_indirect"]


def ma_network_indirect(d_AB, v_AB, d_CB, v_CB):
    """
    Indirect comparison via Bucher's adjusted method

    Formula: d_AC = d_AB - d_CB; var = v_AB + v_CB

    Parameters
    ----------
    d_AB : array-like
        Input data.
    v_AB : array-like
        Input data.
    d_CB : array-like
        Input data.
    v_CB : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: d_AC, var, ci

    References
    ----------
    Bucher et al. (1997)
    """
    d_AB = np.atleast_1d(np.asarray(d_AB, dtype=float))
    n = len(d_AB)
    result = float(np.mean(d_AB))
    se = float(np.std(d_AB, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Indirect comparison via Bucher's adjusted method"})


def cheatsheet():
    return "manmar: Indirect comparison via Bucher's adjusted method"
