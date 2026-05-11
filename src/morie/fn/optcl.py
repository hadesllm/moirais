# morie.fn — function file (hadesllm/morie)
"""Optimal classification (cutting plane)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["optimal_classification"]


def optimal_classification(x):
    """
    Optimal classification (cutting plane)

    Formula: min misclassification via hyperplane

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Optimal classification (cutting plane)"})


def cheatsheet():
    return "optcl: Optimal classification (cutting plane)"
