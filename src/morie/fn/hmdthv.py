# morie.fn -- function file (hadesllm/morie)
"""Decision trees exhibit high variance; small data changes yield very different trees."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tree_high_variance"]


def geron_tree_high_variance(X, y):
    """
    Decision trees exhibit high variance; small data changes yield very different trees

    Formula: Var(f_tree(x)) large relative to Bias^2(f_tree)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance_estimate

    References
    ----------
    Géron Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decision trees exhibit high variance; small data changes yield very different trees"})


def cheatsheet():
    return "hmdthv: Decision trees exhibit high variance; small data changes yield very different trees"
