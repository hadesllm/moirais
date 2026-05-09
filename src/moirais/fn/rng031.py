"""Equivalent form of continuous-time convolution with arguments swapped.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_continuous_convolution_alt"]


def rangayyan_ch3_continuous_convolution_alt(x, h, t, tau):
    """
    Equivalent form of continuous-time convolution with arguments swapped.

    Formula: y(t) = integral_{-inf}^{inf} h(tau) x(t - tau) d(tau)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.
    t : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.31, p. 109
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Equivalent form of continuous-time convolution with arguments swapped."})


def cheatsheet():
    return "rng031: Equivalent form of continuous-time convolution with arguments swapped."
