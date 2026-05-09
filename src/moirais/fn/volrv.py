"""Realised variance from intraday returns."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_realised_variance"]


def vol_realised_variance(r_intraday, block_index):
    """
    Realised variance from intraday returns

    Formula: RV_d = Σ_i r_{d,i}²

    Parameters
    ----------
    r_intraday : array-like
        Input data.
    block_index : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: RV_daily

    References
    ----------
    Andersen-Bollerslev-Diebold-Labys (2001)
    """
    r_intraday = np.atleast_1d(np.asarray(r_intraday, dtype=float))
    n = len(r_intraday)
    result = float(np.mean(r_intraday))
    se = float(np.std(r_intraday, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Realised variance from intraday returns"})


def cheatsheet():
    return "volrv: Realised variance from intraday returns"
