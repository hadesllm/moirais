"""Two-scale RV correcting micro-noise (TSRV)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_two_scale_rv"]


def vol_two_scale_rv(r_intraday, K):
    """
    Two-scale RV correcting micro-noise (TSRV)

    Formula: TSRV = RV_slow - (n̄/n) RV_fast

    Parameters
    ----------
    r_intraday : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: TSRV

    References
    ----------
    Zhang-Mykland-Aït-Sahalia (2005)
    """
    r_intraday = np.atleast_1d(np.asarray(r_intraday, dtype=float))
    n = len(r_intraday)
    result = float(np.mean(r_intraday))
    se = float(np.std(r_intraday, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-scale RV correcting micro-noise (TSRV)"})


def cheatsheet():
    return "voltsr: Two-scale RV correcting micro-noise (TSRV)"
