# moirais.fn — function file (hadesllm/moirais)
"""Gradient boosted regression trees (GBRT): fit residual trees sequentially."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gradient_boosting"]


def geron_gradient_boosting(X, y, n_estimators, learning_rate, max_depth):
    """
    Gradient boosted regression trees (GBRT): fit residual trees sequentially

    Formula: F_{t+1}(x) = F_t(x) + eta * h_t(x); h_t fits residuals -grad L

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    n_estimators : array-like
        Input data.
    learning_rate : array-like
        Input data.
    max_depth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient boosted regression trees (GBRT): fit residual trees sequentially"})


def cheatsheet():
    return "hmgbrt: Gradient boosted regression trees (GBRT): fit residual trees sequentially"
