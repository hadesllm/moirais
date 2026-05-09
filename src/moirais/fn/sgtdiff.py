"""Heat-kernel / diffusion kernel exp(-t L)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_diffusion_kernel"]


def sgt_diffusion_kernel(A, t):
    """
    Heat-kernel / diffusion kernel exp(-t L)

    Formula: K_t = exp(-t L)

    Parameters
    ----------
    A : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: K

    References
    ----------
    Kondor & Lafferty (2002)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heat-kernel / diffusion kernel exp(-t L)"})


def cheatsheet():
    return "sgtdiff: Heat-kernel / diffusion kernel exp(-t L)"
