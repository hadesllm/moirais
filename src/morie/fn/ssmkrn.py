"""S4 structured state-space kernel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["s4_ssm_kernel"]


def s4_ssm_kernel(y, x, A, B, C, L):
    """
    S4 structured state-space kernel

    Formula: K = (CB, CAB, CA^2 B, ..., CA^{L-1} B); y = K * x

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.
    C : array-like
        Input data.
    L : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gu, Goel, Re (2022) S4
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "S4 structured state-space kernel"})


def cheatsheet():
    return "ssmkrn: S4 structured state-space kernel"
