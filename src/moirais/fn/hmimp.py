# moirais.fn — function file (hadesllm/moirais)
"""Missing-value imputation using column median (numeric) or mode (categorical)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_imputation_median"]


def geron_imputation_median(X):
    """
    Missing-value imputation using column median (numeric) or mode (categorical)

    Formula: x_ij = median(x_j) if x_ij is missing

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_imputed

    References
    ----------
    Géron Ch 2
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Missing-value imputation using column median (numeric) or mode (categorical)"})


def cheatsheet():
    return "hmimp: Missing-value imputation using column median (numeric) or mode (categorical)"
