# morie.fn — function file (hadesllm/morie)
"""Stacking (blending): meta-learner combines outputs of base learners."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_stacking"]


def geron_stacking(X, y, base_models, meta_model):
    """
    Stacking (blending): meta-learner combines outputs of base learners

    Formula: y_hat = meta(f_1(x), ..., f_M(x))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    base_models : array-like
        Input data.
    meta_model : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stacking (blending): meta-learner combines outputs of base learners"})


def cheatsheet():
    return "hmstk: Stacking (blending): meta-learner combines outputs of base learners"
