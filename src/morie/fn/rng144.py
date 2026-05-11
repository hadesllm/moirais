"""Gradient of MSE cost function with respect to tap-weight vector.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_mse_gradient"]


def rangayyan_ch3_mse_gradient(w, Theta, Phi):
    """
    Gradient of MSE cost function with respect to tap-weight vector.

    Formula: dJ(w)/dw = -2*Theta + 2*Phi*w

    Parameters
    ----------
    w : array-like
        Input data.
    Theta : array-like
        Input data.
    Phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.167, p. 175
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient of MSE cost function with respect to tap-weight vector."})


def cheatsheet():
    return "rng144: Gradient of MSE cost function with respect to tap-weight vector."
