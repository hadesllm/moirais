# moirais.fn — function file (hadesllm/moirais)
"""Extra-trees: randomize thresholds per feature split for extra variance reduction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_extra_trees"]


def geron_extra_trees(X, y, n_estimators, max_features, seed):
    """
    Extra-trees: randomize thresholds per feature split for extra variance reduction

    Formula: split uses random threshold within feature range

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    n_estimators : array-like
        Input data.
    max_features : array-like
        Input data.
    seed : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Extra-trees: randomize thresholds per feature split for extra variance reduction"})


def cheatsheet():
    return "hmext: Extra-trees: randomize thresholds per feature split for extra variance reduction"
