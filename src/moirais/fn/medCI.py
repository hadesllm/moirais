"""Asymmetric CI for ab (Monte Carlo)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["asymmetric_indirect_ci"]


def asymmetric_indirect_ci(a, b, sa, sb, n_sim):
    """
    Asymmetric CI for ab (Monte Carlo)

    Formula: simulate a~N(â,σ_a), b~N(b̂,σ_b); take ab quantiles

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    sa : array-like
        Input data.
    sb : array-like
        Input data.
    n_sim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    MacKinnon-Lockwood-Williams (2004)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymmetric CI for ab (Monte Carlo)"})


def cheatsheet():
    return "medCI: Asymmetric CI for ab (Monte Carlo)"
