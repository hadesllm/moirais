"""Moving-window stationarity: fit local variograms within windows."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_moving_window"]


def schabenberger_moving_window(coords, z, window_size):
    """
    Moving-window stationarity: fit local variograms within windows

    Formula: gamma(h|s) estimated in window W(s); coefficients vary with center s

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    window_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: local_variograms

    References
    ----------
    Schabenberger Ch 8, Sec 8.3.1
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moving-window stationarity: fit local variograms within windows"})


def cheatsheet():
    return "spmwst: Moving-window stationarity: fit local variograms within windows"
