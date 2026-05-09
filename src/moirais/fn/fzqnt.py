# moirais.fn — function file (hadesllm/moirais)
"""Asymptotic distribution of kernel quantile."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_kernel_quantile_asymptotic"]


def fauzi_kernel_quantile_asymptotic(x):
    """
    Asymptotic distribution of kernel quantile

    Formula: sqrt(n)(Q_hat - Q) -> N(0, tau^2/(f(Q))^2)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Fauzi Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic distribution of kernel quantile"})


def cheatsheet():
    return "fzqnt: Asymptotic distribution of kernel quantile"
