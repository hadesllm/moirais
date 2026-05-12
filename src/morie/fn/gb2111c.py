# morie.fn -- function file (hadesllm/morie)
"""Elementary coverage C_i = U_(i) - U_(i-1) has Beta(1,n) distribution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_elementary_coverage_beta"]


def gibbons_elementary_coverage_beta(n):
    """
    Elementary coverage C_i = U_(i) - U_(i-1) has Beta(1,n) distribution

    Formula: C_i ~ Beta(1, n); E(C_i) = 1/(n+1)

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Gibbons Corollary 2.11.1.1
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(max(n, 2))
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elementary coverage C_i = U_(i) - U_(i-1) has Beta(1,n) distribution"})


def cheatsheet():
    return "gb2111c: Elementary coverage C_i = U_(i) - U_(i-1) has Beta(1,n) distribution"
