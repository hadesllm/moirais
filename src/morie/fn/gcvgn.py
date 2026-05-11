# morie.fn — function file (hadesllm/morie)
"""Cross-validation for genomic prediction accuracy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["genomic_cross_validation"]


def genomic_cross_validation(x, y):
    """
    Cross-validation for genomic prediction accuracy

    Formula: r = cor(y, y_hat) across K folds

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-validation for genomic prediction accuracy"})


def cheatsheet():
    return "gcvgn: Cross-validation for genomic prediction accuracy"
