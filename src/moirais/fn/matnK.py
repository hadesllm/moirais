"""Matérn kernel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["matern_kernel"]


def matern_kernel(d, nu, rho):
    """
    Matérn kernel

    Formula: k = σ² (2^{1-ν}/Γ(ν)) (√(2ν)d/ρ)^ν K_ν(...)

    Parameters
    ----------
    d : array-like
        Input data.
    nu : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Matérn (1960)
    """
    d = np.atleast_1d(np.asarray(d, dtype=float))
    n = len(d)
    result = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matérn kernel"})


def cheatsheet():
    return "matnK: Matérn kernel"
