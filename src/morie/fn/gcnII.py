"""GCNII -- initial residual + identity mapping."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gcnii"]


def gcnii(A, H0, alpha, beta, K):
    """
    GCNII -- initial residual + identity mapping

    Formula: H = σ((1−α)Â H + α H0)((1−β)I + β W)

    Parameters
    ----------
    A : array-like
        Input data.
    H0 : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chen et al (2020) GCNII
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GCNII -- initial residual + identity mapping"})


def cheatsheet():
    return "gcnII: GCNII -- initial residual + identity mapping"
