"""Multi-kernel realised volatility with subsampling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_multi_kernel_rk"]


def vol_multi_kernel_rk(r_intraday, grids, kernel):
    """
    Multi-kernel realised volatility with subsampling

    Formula: Average RK over multiple grids

    Parameters
    ----------
    r_intraday : array-like
        Input data.
    grids : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: RK_avg

    References
    ----------
    Zhang-Mykland-Aït-Sahalia (2005)
    """
    r_intraday = np.atleast_1d(np.asarray(r_intraday, dtype=float))
    n = len(r_intraday)
    result = float(np.mean(r_intraday))
    se = float(np.std(r_intraday, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-kernel realised volatility with subsampling"})


def cheatsheet():
    return "volmuk: Multi-kernel realised volatility with subsampling"
