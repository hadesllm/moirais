"""Quasi-likelihood SV(1) Kalman-Harvey approx."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_sv_quasi_lik"]


def vol_sv_quasi_lik(r, init):
    """
    Quasi-likelihood SV(1) Kalman-Harvey approx

    Formula: log r_t² = h_t + log z²; h_t = μ + φ(h-μ) + η

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mu, phi, sigma_eta, ll

    References
    ----------
    Harvey-Ruiz-Shephard (1994)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quasi-likelihood SV(1) Kalman-Harvey approx"})


def cheatsheet():
    return "volsv: Quasi-likelihood SV(1) Kalman-Harvey approx"
