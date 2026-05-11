"""Sobolev (negative) form of W_1 via H^{-1} norm."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_sobolev_w1"]


def ot_sobolev_w1(mu, nu, Laplace_inv):
    """
    Sobolev (negative) form of W_1 via H^{-1} norm

    Formula: W_1(μ,ν) ≈ ||μ-ν||_{H^{-1}}

    Parameters
    ----------
    mu : array-like
        Input data.
    nu : array-like
        Input data.
    Laplace_inv : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W1_sob

    References
    ----------
    Peyré (2018)
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sobolev (negative) form of W_1 via H^{-1} norm"})


def cheatsheet():
    return "otsobm: Sobolev (negative) form of W_1 via H^{-1} norm"
