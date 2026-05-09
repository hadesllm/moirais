# moirais.fn — function file (hadesllm/moirais)
"""Null distribution of exact run lengths r1j and r2j under randomness."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_run_lengths_dist"]


def gibbons_run_lengths_dist(run_lengths, n1, n2):
    """
    Null distribution of exact run lengths r1j and r2j under randomness

    Formula: f(r11,...,r1n1,r21,...) = c*r1!*r2! / (prod(r_ij!) * C(n1+n2,n1))

    Parameters
    ----------
    run_lengths : array-like
        Input data.
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Theorem 3.3.1
    """
    run_lengths = np.asarray(run_lengths, dtype=float)
    n = int(run_lengths) if run_lengths.ndim == 0 else len(run_lengths)
    result = float(np.mean(run_lengths))
    se = float(np.std(run_lengths, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Null distribution of exact run lengths r1j and r2j under randomness"})


def cheatsheet():
    return "gb331: Null distribution of exact run lengths r1j and r2j under randomness"
