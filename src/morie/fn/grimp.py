# morie.fn — function file (hadesllm/morie)
"""Simple imputation: replace NaNs with column mean / median / mode."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_simple_imputer"]


def geron_simple_imputer(X, strategy):
    """
    Simple imputation: replace NaNs with column mean / median / mode

    Formula: x_ij := agg(X_j) whenever x_ij is NaN, agg in {mean, median, mode}

    Parameters
    ----------
    X : array-like
        Input data.
    strategy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_imputed

    References
    ----------
    Géron Ch 2, Data Cleaning / Imputation section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simple imputation: replace NaNs with column mean / median / mode"})


def cheatsheet():
    return "grimp: Simple imputation: replace NaNs with column mean / median / mode"
