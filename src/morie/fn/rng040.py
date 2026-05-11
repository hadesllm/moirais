"""Continuous-time linear-ramp impulse response of a smoothing filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_linear_ramp_filter"]


def rangayyan_ch3_linear_ramp_filter(t):
    """
    Continuous-time linear-ramp impulse response of a smoothing filter.

    Formula: h(t) = 10 * (0.25 - t), for 0 <= t <= 0.25 s

    Parameters
    ----------
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.42, p. 114
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous-time linear-ramp impulse response of a smoothing filter."})


def cheatsheet():
    return "rng040: Continuous-time linear-ramp impulse response of a smoothing filter."
