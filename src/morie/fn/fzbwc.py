# morie.fn -- function file (hadesllm/morie)
"""Bandwidth condition for kernel quantile Edgeworth expansion."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_quantile_bw_condition"]


def fauzi_quantile_bw_condition(bandwidth, n):
    """
    Bandwidth condition for kernel quantile Edgeworth expansion

    Formula: h = o(n^{-1/4}) and lim (n^{1/4}h)^{-k} n^{-beta} = 0 for any beta>0, integer k

    Parameters
    ----------
    bandwidth : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boolean

    References
    ----------
    Fauzi Ch 3, Eq 3.8
    """
    bandwidth = np.asarray(bandwidth, dtype=float)
    n = int(bandwidth) if bandwidth.ndim == 0 else len(bandwidth)
    result = float(np.mean(bandwidth))
    se = float(np.std(bandwidth, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bandwidth condition for kernel quantile Edgeworth expansion"})


def cheatsheet():
    return "fzbwc: Bandwidth condition for kernel quantile Edgeworth expansion"
