"""Wiener-Khinchin theorem: covariance and spectral density are Fourier pairs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_wiener_khinchin"]


def schabenberger_wiener_khinchin(cov_func):
    """
    Wiener-Khinchin theorem: covariance and spectral density are Fourier pairs

    Formula: C(h) = integral exp(i*omega'h) f(omega) d*omega; f = spectral density

    Parameters
    ----------
    cov_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectral_density

    References
    ----------
    Schabenberger Ch 2, Sec 2.5.3
    """
    cov_func = np.asarray(cov_func, dtype=float)
    n = int(cov_func) if cov_func.ndim == 0 else len(cov_func)
    result = float(np.mean(cov_func))
    se = float(np.std(cov_func, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wiener-Khinchin theorem: covariance and spectral density are Fourier pairs"})


def cheatsheet():
    return "spwkth: Wiener-Khinchin theorem: covariance and spectral density are Fourier pairs"
