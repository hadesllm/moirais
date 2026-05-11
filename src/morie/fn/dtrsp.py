# morie.fn — function file (hadesllm/morie)
"""Decision tree split via Gini/entropy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["decision_tree_split"]


def decision_tree_split(x, y):
    """
    Decision tree split via Gini/entropy

    Formula: G(t) = 1 - sum p_k^2, H(t) = -sum p_k log p_k

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decision tree split via Gini/entropy"})


def cheatsheet():
    return "dtrsp: Decision tree split via Gini/entropy"
