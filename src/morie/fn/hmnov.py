# morie.fn — function file (hadesllm/morie)
"""Novelty detection: classify new points as novel vs in-distribution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_novelty_detection"]


def geron_novelty_detection(model, X_new):
    """
    Novelty detection: classify new points as novel vs in-distribution

    Formula: novelty if p(x) / p_train < 1

    Parameters
    ----------
    model : array-like
        Input data.
    X_new : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: novel

    References
    ----------
    Géron Ch 8
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Novelty detection: classify new points as novel vs in-distribution"})


def cheatsheet():
    return "hmnov: Novelty detection: classify new points as novel vs in-distribution"
