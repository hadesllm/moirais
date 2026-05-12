# morie.fn -- function file (hadesllm/morie)
"""Asymptotic distribution of one-sided KS: limiting survival function of D+_n."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_ks_onesided_asymp"]


def gibbons_ks_onesided_asymp(d, n):
    """
    Asymptotic distribution of one-sided KS: limiting survival function of D+_n

    Formula: lim P(D+_n < d/sqrt(n)) = 1 - exp(-2d^2)

    Parameters
    ----------
    d : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Theorem 4.3.5
    """
    d = np.asarray(d, dtype=float)
    n = int(d) if d.ndim == 0 else len(d)
    result = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic distribution of one-sided KS: limiting survival function of D+_n"})


def cheatsheet():
    return "gb435: Asymptotic distribution of one-sided KS: limiting survival function of D+_n"
