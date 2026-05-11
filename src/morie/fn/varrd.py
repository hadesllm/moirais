"""Variance reduction criterion for regression tree splitting."""
import numpy as np
from ._richresult import RichResult

__all__ = ["variance_reduction_split"]


def variance_reduction_split(y, split_idx):
    """
    Variance reduction criterion for regression tree splitting

    Formula: Delta_Var = Var(t) - (n_L/n)*Var(t_L) - (n_R/n)*Var(t_R)

    Parameters
    ----------
    y : array-like
        Input data.
    split_idx : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'delta_var': 'float'}

    References
    ----------
    Montesinos Lopez Ch 15
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance reduction criterion for regression tree splitting"})


def cheatsheet():
    return "varrd: Variance reduction criterion for regression tree splitting"
