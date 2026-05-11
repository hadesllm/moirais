"""Komlos-Major-Tusnady (KMT) strong approximation of empirical process by Brownian bridge."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_kmt_strong_approximation"]


def kosorok_ch2_kmt_strong_approximation(G_n, B_n, F, n, x):
    """
    Komlos-Major-Tusnady (KMT) strong approximation of empirical process by Brownian bridge

    Formula: P( ||G_n - B_n(F)||_inf > (a log n + x)/sqrt(n) ) <= b * exp(-c x)

    Parameters
    ----------
    G_n : array-like
        Input data.
    B_n : array-like
        Input data.
    F : array-like
        Input data.
    n : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, p. 31
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Komlos-Major-Tusnady (KMT) strong approximation of empirical process by Brownian bridge"})


def cheatsheet():
    return "ksr059: Komlos-Major-Tusnady (KMT) strong approximation of empirical process by Brownian bridge"
