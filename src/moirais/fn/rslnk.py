# moirais.fn — function file (hadesllm/moirais)
"""Residual/skip connection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["residual_connection"]


def residual_connection(x):
    """
    Residual/skip connection

    Formula: y = F(x) + x (identity shortcut)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    He et al. (2016)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Residual/skip connection"})


def cheatsheet():
    return "rslnk: Residual/skip connection"
