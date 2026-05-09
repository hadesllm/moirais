# moirais.fn — function file (hadesllm/moirais)
"""Relative MAE: MAE of target model / MAE of baseline."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_relative_mae"]


def joseph_relative_mae(y_true, y_pred, y_baseline):
    """
    Relative MAE: MAE of target model / MAE of baseline

    Formula: RelMAE = MAE(y, y_hat) / MAE(y, y_hat_baseline)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.
    y_baseline : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rel_mae

    References
    ----------
    Joseph Ch 19, RelMAE section
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Relative MAE: MAE of target model / MAE of baseline"})


def cheatsheet():
    return "jorelm: Relative MAE: MAE of target model / MAE of baseline"
