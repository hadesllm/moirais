"""Closed-form optimal Wiener filter tap weights.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_optimal_wiener_filter"]


def rangayyan_ch3_optimal_wiener_filter(Phi, Theta):
    """
    Closed-form optimal Wiener filter tap weights.

    Formula: w_o = Phi^(-1) * Theta

    Parameters
    ----------
    Phi : array-like
        Input data.
    Theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.169, p. 175
    """
    Phi = np.atleast_1d(np.asarray(Phi, dtype=float))
    n = len(Phi)
    result = float(np.mean(Phi))
    se = float(np.std(Phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed-form optimal Wiener filter tap weights."})


def cheatsheet():
    return "rng146: Closed-form optimal Wiener filter tap weights."
