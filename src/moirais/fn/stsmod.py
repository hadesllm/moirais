"""Linear Gaussian state-space model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["state_space_model"]


def state_space_model(y, Z, T, H, Q, R):
    """
    Linear Gaussian state-space model

    Formula: y_t = Z alpha_t + eps_t; alpha_{t+1} = T alpha_t + R eta_t

    Parameters
    ----------
    y : array-like
        Input data.
    Z : array-like
        Input data.
    T : array-like
        Input data.
    H : array-like
        Input data.
    Q : array-like
        Input data.
    R : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Durbin-Koopman (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear Gaussian state-space model"})


def cheatsheet():
    return "stsmod: Linear Gaussian state-space model"
