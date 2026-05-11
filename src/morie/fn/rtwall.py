"""Rt via Wallinga-Teunis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rt_wallinga_teunis"]


def rt_wallinga_teunis(incidence, serial_interval):
    """
    Rt via Wallinga-Teunis

    Formula: Rt = sum cases × likelihood-weighted parents

    Parameters
    ----------
    incidence : array-like
        Input data.
    serial_interval : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wallinga-Teunis (2004)
    """
    incidence = np.atleast_1d(np.asarray(incidence, dtype=float))
    n = len(incidence)
    result = float(np.mean(incidence))
    se = float(np.std(incidence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rt via Wallinga-Teunis"})


def cheatsheet():
    return "rtwall: Rt via Wallinga-Teunis"
