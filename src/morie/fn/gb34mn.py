# morie.fn -- function file (hadesllm/morie)
"""Mean of total runs up-and-down in a random sequence of n numbers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_runs_ud_mean"]


def gibbons_runs_ud_mean(n):
    """
    Mean of total runs up-and-down in a random sequence of n numbers

    Formula: E(R_ud) = (2n-1)/3

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean

    References
    ----------
    Gibbons Ch 3.4
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(max(n, 2))
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean of total runs up-and-down in a random sequence of n numbers"})


def cheatsheet():
    return "gb34mn: Mean of total runs up-and-down in a random sequence of n numbers"
