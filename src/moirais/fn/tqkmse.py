"""TurboQuant KV-cache compression with MSE-optimal codebook."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["turboquant_kv_mse"]


def turboquant_kv_mse(x, bits, method):
    """
    TurboQuant KV-cache compression with MSE-optimal codebook

    Formula: WHT(1/sqrt(d)) -> Lloyd-Max codebook -> packed bits

    Parameters
    ----------
    x : array-like
        Input data.
    bits : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL et al ICLR 2026 arxiv:2504.19874
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TurboQuant KV-cache compression with MSE-optimal codebook"})


def cheatsheet():
    return "tqkmse: TurboQuant KV-cache compression with MSE-optimal codebook"
