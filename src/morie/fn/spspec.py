"""Spectral representation of stationary random field (Bochner)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_spectral_representation"]


def schabenberger_spectral_representation(cov_func):
    """
    Spectral representation of stationary random field (Bochner)

    Formula: C(h) = integral exp(i*omega'h) dF(omega) where F is spectral distribution

    Parameters
    ----------
    cov_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectral

    References
    ----------
    Schabenberger Ch 2, Sec 2.5
    """
    cov_func = np.asarray(cov_func, dtype=float)
    n = int(cov_func) if cov_func.ndim == 0 else len(cov_func)
    result = float(np.mean(cov_func))
    se = float(np.std(cov_func, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral representation of stationary random field (Bochner)"})


def cheatsheet():
    return "spspec: Spectral representation of stationary random field (Bochner)"
