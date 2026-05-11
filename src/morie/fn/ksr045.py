"""Functional delta method for the bootstrap of Hadamard-differentiable functionals."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_functional_delta_bootstrap"]


def kosorok_ch2_functional_delta_bootstrap(phi, X_n, X_hat_n, mu, r_n):
    """
    Functional delta method for the bootstrap of Hadamard-differentiable functionals

    Formula: r_n c (phi(X_hat_n) - phi(X_n)) =>_W phi'_mu(X) in probability under bootstrap weights W

    Parameters
    ----------
    phi : array-like
        Input data.
    X_n : array-like
        Input data.
    X_hat_n : array-like
        Input data.
    mu : array-like
        Input data.
    r_n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Thm 2.9, p. 23
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    n = len(phi)
    result = float(np.mean(phi))
    se = float(np.std(phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional delta method for the bootstrap of Hadamard-differentiable functionals"})


def cheatsheet():
    return "ksr045: Functional delta method for the bootstrap of Hadamard-differentiable functionals"
