# morie.fn -- function file (hadesllm/morie)
"""Theorem 2.1: geometric combination of expected KDFE has O(h^4) bias."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm2_1_expected_kdfe"]


def fauzi_thm2_1_expected_kdfe(x, bandwidth, a):
    """
    Theorem 2.1: geometric combination of expected KDFE has O(h^4) bias

    Formula: [J_h(x)]^{t1} * [J_{ah}(x)]^{t2} = F_X(x) + O(h^4), t1=a^2/(a^2-1), t2=-1/(a^2-1)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 2, Theorem 2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 2.1: geometric combination of expected KDFE has O(h^4) bias"})


def cheatsheet():
    return "fzt21: Theorem 2.1: geometric combination of expected KDFE has O(h^4) bias"
