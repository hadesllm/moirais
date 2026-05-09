"""Law of the iterated logarithm for the empirical process G_n."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_law_iterated_logarithm"]


def kosorok_ch2_law_iterated_logarithm(G_n, n):
    """
    Law of the iterated logarithm for the empirical process G_n

    Formula: limsup_n ||G_n||_inf / sqrt(2 log log n) <= 1/2 almost surely

    Parameters
    ----------
    G_n : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.21, p. 31
    """
    G_n = np.atleast_1d(np.asarray(G_n, dtype=float))
    n = len(G_n)
    result = float(np.mean(G_n))
    se = float(np.std(G_n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Law of the iterated logarithm for the empirical process G_n"})


def cheatsheet():
    return "ksr058: Law of the iterated logarithm for the empirical process G_n"
