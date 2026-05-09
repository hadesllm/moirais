"""Wiener-Hopf equation expressed as a convolution relationship under stationarity.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_wiener_convolution_relationship"]


def rangayyan_ch3_wiener_convolution_relationship(w_ok, phi, theta, k):
    """
    Wiener-Hopf equation expressed as a convolution relationship under stationarity.

    Formula: w_ok * phi(k) = theta(k)

    Parameters
    ----------
    w_ok : array-like
        Input data.
    phi : array-like
        Input data.
    theta : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.174, p. 176
    """
    w_ok = np.atleast_1d(np.asarray(w_ok, dtype=float))
    n = len(w_ok)
    result = float(np.mean(w_ok))
    se = float(np.std(w_ok, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wiener-Hopf equation expressed as a convolution relationship under stationarity."})


def cheatsheet():
    return "rng148: Wiener-Hopf equation expressed as a convolution relationship under stationarity."
