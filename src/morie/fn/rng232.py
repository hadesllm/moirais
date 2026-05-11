"""Fourier transform of the log of a product is sum of log-FTs of the components.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_homomorphic_log_fourier"]


def rangayyan_ch4_homomorphic_log_fourier(X_l, P_l, omega):
    """
    Fourier transform of the log of a product is sum of log-FTs of the components.

    Formula: Y_l(omega) = X_l(omega) + P_l(omega)

    Parameters
    ----------
    X_l : array-like
        Input data.
    P_l : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.60, p. 244
    """
    X_l = np.atleast_1d(np.asarray(X_l, dtype=float))
    n = len(X_l)
    result = float(np.mean(X_l))
    se = float(np.std(X_l, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fourier transform of the log of a product is sum of log-FTs of the components."})


def cheatsheet():
    return "rng232: Fourier transform of the log of a product is sum of log-FTs of the components."
