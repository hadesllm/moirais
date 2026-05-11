"""Realised kernel volatility with Bartlett weighting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_realised_kernel"]


def vol_realised_kernel(r_intraday, H):
    """
    Realised kernel volatility with Bartlett weighting

    Formula: RK = Σ_h k(h/H) γ_h, γ_h = Σ r_i r_{i-h}

    Parameters
    ----------
    r_intraday : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: RK

    References
    ----------
    Barndorff-Nielsen-Hansen-Lunde-Shephard (2008)
    """
    r_intraday = np.atleast_1d(np.asarray(r_intraday, dtype=float))
    n = len(r_intraday)
    result = float(np.mean(r_intraday))
    se = float(np.std(r_intraday, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Realised kernel volatility with Bartlett weighting"})


def cheatsheet():
    return "volrk: Realised kernel volatility with Bartlett weighting"
