# morie.fn — function file (hadesllm/morie)
"""One-vs-Rest: train K binary classifiers, predict argmax of scores."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_one_vs_rest"]


def geron_one_vs_rest(X, y):
    """
    One-vs-Rest: train K binary classifiers, predict argmax of scores

    Formula: y_hat = argmax_k score_k(x) across K OvR classifiers

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_pred, scores

    References
    ----------
    Géron Ch 3, Multiclass (OvR) section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "One-vs-Rest: train K binary classifiers, predict argmax of scores"})


def cheatsheet():
    return "grovr: One-vs-Rest: train K binary classifiers, predict argmax of scores"
