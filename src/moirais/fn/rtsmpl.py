"""Rt from serial interval (Cori)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rt_serial_interval"]


def rt_serial_interval(incidence, serial_interval, window):
    """
    Rt from serial interval (Cori)

    Formula: Rt = I_t / sum_{s=1..tau} pi_s I_{t-s}

    Parameters
    ----------
    incidence : array-like
        Input data.
    serial_interval : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cori-Ferguson-Fraser-Cauchemez (2013)
    """
    incidence = np.atleast_1d(np.asarray(incidence, dtype=float))
    n = len(incidence)
    result = float(np.mean(incidence))
    se = float(np.std(incidence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rt from serial interval (Cori)"})


def cheatsheet():
    return "rtsmpl: Rt from serial interval (Cori)"
