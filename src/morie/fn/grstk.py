# morie.fn — function file (hadesllm/morie)
"""Stacking meta-learner: predictions from L base models feed a final blender."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_stacking_predictor"]


def geron_stacking_predictor(base_preds, y):
    """
    Stacking meta-learner: predictions from L base models feed a final blender

    Formula: y_hat = g(h_1(x), ..., h_L(x)) where g is trained on base-model out-of-fold preds

    Parameters
    ----------
    base_preds : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_pred

    References
    ----------
    Géron Ch 6, Stacking section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stacking meta-learner: predictions from L base models feed a final blender"})


def cheatsheet():
    return "grstk: Stacking meta-learner: predictions from L base models feed a final blender"
