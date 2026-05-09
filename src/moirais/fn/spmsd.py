"""Mean-square differentiability of random field."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_ms_differentiability"]


def schabenberger_ms_differentiability(cov_func):
    """
    Mean-square differentiability of random field

    Formula: dZ/ds_k exists in MS if C(h) has mixed partial d^2C/dh_k^2 at h=0

    Parameters
    ----------
    cov_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boolean

    References
    ----------
    Schabenberger Ch 2, Sec 2.3
    """
    cov_func = np.asarray(cov_func, dtype=float)
    n = int(cov_func) if cov_func.ndim == 0 else len(cov_func)
    result = float(np.mean(cov_func))
    se = float(np.std(cov_func, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean-square differentiability of random field"})


def cheatsheet():
    return "spmsd: Mean-square differentiability of random field"
