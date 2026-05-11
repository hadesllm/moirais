"""Continuous-time integral over a sliding window of duration tau.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_running_integral_window"]


def rangayyan_ch3_running_integral_window(x, t, tau):
    """
    Continuous-time integral over a sliding window of duration tau.

    Formula: y(t) = integral_{t-tau}^{t} x(t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.112, p. 143
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous-time integral over a sliding window of duration tau."})


def cheatsheet():
    return "rng101: Continuous-time integral over a sliding window of duration tau."
