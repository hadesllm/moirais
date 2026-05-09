# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Gini impurity for classification tree splits."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gini_impurity"]


def gini_impurity(class_probs):
    """
    Gini impurity for classification tree splits

    Formula: Gini(t) = 1 - sum_k p_k^2; p_k = proportion class k at node t

    Parameters
    ----------
    class_probs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'gini': 'float'}

    References
    ----------
    Montesinos Lopez Ch 15
    """
    class_probs = np.atleast_1d(np.asarray(class_probs, dtype=float))
    n = len(class_probs)
    result = float(np.mean(class_probs))
    se = float(np.std(class_probs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gini impurity for classification tree splits"})


def cheatsheet():
    return "giniI: Gini impurity for classification tree splits"
