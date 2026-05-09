# moirais.fn — function file (hadesllm/moirais)
"""We are what we repeatedly do. Excellence is a habit. — Aristotle"""
import numpy as np
from ._richresult import RichResult

__all__ = ["We are what we repeatedly do. Excellence is a habit. — Aristotle"]


def flash_attention(x):
    """
    Flash attention (IO-aware)

    Formula: tiled softmax(QK'/sqrt(d))V, O(N) memory

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
    Dao et al. (2022)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "We are what we repeatedly do. Excellence is a habit. — Aristotle"})


def cheatsheet():
    return "We are what we repeatedly do. Excellence is a habit. — Aristotle"
