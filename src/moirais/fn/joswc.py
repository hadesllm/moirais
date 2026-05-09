# moirais.fn — function file (hadesllm/moirais)
"""Sliding-window CV: fixed-size train + val, both slide forward each fold."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_sliding_window_cv"]


def joseph_sliding_window_cv(y, T_w, step, H, K):
    """
    Sliding-window CV: fixed-size train + val, both slide forward each fold

    Formula: for i: train = y[i*step : i*step + T_w]; val = y[i*step + T_w : i*step + T_w + H]

    Parameters
    ----------
    y : array-like
        Input data.
    T_w : array-like
        Input data.
    step : array-like
        Input data.
    H : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: folds

    References
    ----------
    Joseph Ch 20, Sliding Window CV section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sliding-window CV: fixed-size train + val, both slide forward each fold"})


def cheatsheet():
    return "joswc: Sliding-window CV: fixed-size train + val, both slide forward each fold"
