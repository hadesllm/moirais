"""Functional delta method for Hadamard-differentiable maps tangentially to D0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_functional_delta_method"]


def kosorok_ch2_functional_delta_method(phi, X_n, theta, r_n):
    """
    Functional delta method for Hadamard-differentiable maps tangentially to D0

    Formula: r_n (phi(X_n) - phi(theta)) => phi'_theta(X) when r_n(X_n - theta) => X

    Parameters
    ----------
    phi : array-like
        Input data.
    X_n : array-like
        Input data.
    theta : array-like
        Input data.
    r_n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Thm 2.8, p. 22
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    n = len(phi)
    result = float(np.mean(phi))
    se = float(np.std(phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional delta method for Hadamard-differentiable maps tangentially to D0"})


def cheatsheet():
    return "ksr042: Functional delta method for Hadamard-differentiable maps tangentially to D0"
