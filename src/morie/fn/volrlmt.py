"""AR(1) on log realised volatility (HAR alternative)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_realised_log_vol_ar"]


def vol_realised_log_vol_ar(RV):
    """
    AR(1) on log realised volatility (HAR alternative)

    Formula: log RV_t = c + φ log RV_{t-1} + u

    Parameters
    ----------
    RV : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: c, phi, sigma

    References
    ----------
    Andersen-Bollerslev-Diebold (2003)
    """
    RV = np.atleast_1d(np.asarray(RV, dtype=float))
    n = len(RV)
    result = float(np.mean(RV))
    se = float(np.std(RV, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AR(1) on log realised volatility (HAR alternative)"})


def cheatsheet():
    return "volrlmt: AR(1) on log realised volatility (HAR alternative)"
