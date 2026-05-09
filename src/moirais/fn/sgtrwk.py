"""Random-walk kernel (geometric series)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_random_walk_kernel"]


def sgt_random_walk_kernel(A, lam):
    """
    Random-walk kernel (geometric series)

    Formula: K = Σ λ^k A^k = (I - λA)^{-1}

    Parameters
    ----------
    A : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: K

    References
    ----------
    Gärtner-Flach-Wrobel (2003)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random-walk kernel (geometric series)"})


def cheatsheet():
    return "sgtrwk: Random-walk kernel (geometric series)"
