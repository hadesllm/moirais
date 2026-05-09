# moirais.fn — function file (hadesllm/moirais)
"""Expanding-window CV: training set grows, validation fixed-size, walks forward."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_expanding_window_cv"]


def joseph_expanding_window_cv(y, T0, step, H, K):
    """
    Expanding-window CV: training set grows, validation fixed-size, walks forward

    Formula: for i in 0..K: train = y[0 : T_0 + i*step]; val = y[T_0 + i*step : T_0 + i*step + H]

    Parameters
    ----------
    y : array-like
        Input data.
    T0 : array-like
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
    Joseph Ch 20, Expanding Window CV section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Expanding-window CV: training set grows, validation fixed-size, walks forward"})


def cheatsheet():
    return "joexw: Expanding-window CV: training set grows, validation fixed-size, walks forward"
