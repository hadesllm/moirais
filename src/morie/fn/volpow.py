"""Power variation of order p of intraday returns."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_power_variation"]


def vol_power_variation(r_intraday, p):
    """
    Power variation of order p of intraday returns

    Formula: PV_p = Σ |r_i|^p

    Parameters
    ----------
    r_intraday : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: PV

    References
    ----------
    Barndorff-Nielsen-Shephard (2002)
    """
    r_intraday = np.atleast_1d(np.asarray(r_intraday, dtype=float))
    n = len(r_intraday)
    result = float(np.mean(r_intraday))
    se = float(np.std(r_intraday, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Power variation of order p of intraday returns"})


def cheatsheet():
    return "volpow: Power variation of order p of intraday returns"
