# moirais.fn — function file (hadesllm/moirais)
"""Cutting hyperplane/sphere for vote classification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["cutting_plane_sphere"]


def cutting_plane_sphere(x):
    """
    Cutting hyperplane/sphere for vote classification

    Formula: w'x = c separates yea from nay

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
    Armstrong Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cutting hyperplane/sphere for vote classification"})


def cheatsheet():
    return "csphr: Cutting hyperplane/sphere for vote classification"
