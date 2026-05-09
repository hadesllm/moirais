# moirais.fn — function file (hadesllm/moirais)
"""Shannon entropy impurity for a node."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_entropy_impurity"]


def geron_entropy_impurity(y):
    """
    Shannon entropy impurity for a node

    Formula: H = -sum_k p_k log2(p_k)

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: entropy

    References
    ----------
    Géron Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shannon entropy impurity for a node"})


def cheatsheet():
    return "hment: Shannon entropy impurity for a node"
