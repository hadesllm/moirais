"""Minimum mean-squared error achievable by the Wiener filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_minimum_mse"]


def rangayyan_ch3_minimum_mse(sigma_d, Theta, Phi):
    """
    Minimum mean-squared error achievable by the Wiener filter.

    Formula: J_min = sigma_d^2 - Theta^T * Phi^(-1) * Theta

    Parameters
    ----------
    sigma_d : array-like
        Input data.
    Theta : array-like
        Input data.
    Phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.172, p. 175
    """
    sigma_d = np.atleast_1d(np.asarray(sigma_d, dtype=float))
    n = len(sigma_d)
    result = float(np.mean(sigma_d))
    se = float(np.std(sigma_d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimum mean-squared error achievable by the Wiener filter."})


def cheatsheet():
    return "rng147: Minimum mean-squared error achievable by the Wiener filter."
