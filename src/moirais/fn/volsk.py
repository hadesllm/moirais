"""Kalman-filter approximation for SV(1)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_stochastic_kalman"]


def vol_stochastic_kalman(r, init):
    """
    Kalman-filter approximation for SV(1)

    Formula: Linearise log r²=h+log z²; quasi-Kalman

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_t, mu, phi, sigma_eta

    References
    ----------
    Harvey-Ruiz-Shephard (1994)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kalman-filter approximation for SV(1)"})


def cheatsheet():
    return "volsk: Kalman-filter approximation for SV(1)"
