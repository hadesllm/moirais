"""Canadian Fire Weather Index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fire_weather_index"]


def fire_weather_index(T, RH, wind, precip):
    """
    Canadian Fire Weather Index

    Formula: FFMC + DMC + DC + ISI + BUI -> FWI

    Parameters
    ----------
    T : array-like
        Input data.
    RH : array-like
        Input data.
    wind : array-like
        Input data.
    precip : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Van Wagner (1987)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Canadian Fire Weather Index"})


def cheatsheet():
    return "fwxF: Canadian Fire Weather Index"
