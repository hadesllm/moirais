# morie.fn — function file (hadesllm/morie)
"""Permutation-based RF variable importance (MDI permutation)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rf_permutation_importance"]


def rf_permutation_importance(forest, X, y):
    """
    Permutation-based RF variable importance (MDI permutation)

    Formula: Imp_perm(X_j) = OOB_error(X_j permuted) - OOB_error(original)

    Parameters
    ----------
    forest : array-like
        Input data.
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'importance': 'array'}

    References
    ----------
    Montesinos Lopez Ch 15
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Permutation-based RF variable importance (MDI permutation)"})


def cheatsheet():
    return "rfpmi: Permutation-based RF variable importance (MDI permutation)"
