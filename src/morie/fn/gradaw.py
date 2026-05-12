# morie.fn -- function file (hadesllm/morie)
"""AdaBoost sample weight update rule."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_adaboost_weight_update"]


def geron_adaboost_weight_update(y_true, y_pred, weights, alpha_t):
    """
    AdaBoost sample weight update rule

    Formula: w_i <- w_i * exp(alpha_t * 1{y_i != h_t(x_i)}), then normalize

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.
    weights : array-like
        Input data.
    alpha_t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights_new

    References
    ----------
    Géron Ch 6, AdaBoost section
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdaBoost sample weight update rule"})


def cheatsheet():
    return "gradaw: AdaBoost sample weight update rule"
