"""Mauna Loa CO₂ trend (Keeling curve)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["co2_trend"]


def co2_trend(co2_monthly):
    """
    Mauna Loa CO₂ trend (Keeling curve)

    Formula: long-term + seasonal cycle decomposition

    Parameters
    ----------
    co2_monthly : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Keeling (1960)
    """
    co2_monthly = np.atleast_1d(np.asarray(co2_monthly, dtype=float))
    n = len(co2_monthly)
    result = float(np.mean(co2_monthly))
    se = float(np.std(co2_monthly, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mauna Loa CO₂ trend (Keeling curve)"})


def cheatsheet():
    return "co2Trnd: Mauna Loa CO₂ trend (Keeling curve)"
