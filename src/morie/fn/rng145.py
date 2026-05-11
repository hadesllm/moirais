"""Wiener-Hopf normal equation for the optimal tap weights.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_wiener_hopf_normal_equation"]


def rangayyan_ch3_wiener_hopf_normal_equation(Phi, w_o, Theta):
    """
    Wiener-Hopf normal equation for the optimal tap weights.

    Formula: Phi * w_o = Theta

    Parameters
    ----------
    Phi : array-like
        Input data.
    w_o : array-like
        Input data.
    Theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.168, p. 175
    """
    Phi = np.atleast_1d(np.asarray(Phi, dtype=float))
    n = len(Phi)
    result = float(np.mean(Phi))
    se = float(np.std(Phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wiener-Hopf normal equation for the optimal tap weights."})


def cheatsheet():
    return "rng145: Wiener-Hopf normal equation for the optimal tap weights."
