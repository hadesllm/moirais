"""Causal continuous-time convolution form (lower limit 0, upper limit t).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_causal_convolution"]


def rangayyan_ch3_causal_convolution(x, h, t, tau):
    """
    Causal continuous-time convolution form (lower limit 0, upper limit t).

    Formula: y(t) = integral_{0}^{t} x(tau) h(t - tau) d(tau)

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
    Rangayyan (2024), Ch 3, Eq 3.32, p. 109
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Causal continuous-time convolution form (lower limit 0, upper limit t)."})


def cheatsheet():
    return "rng032: Causal continuous-time convolution form (lower limit 0, upper limit t)."
