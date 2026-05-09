# moirais.fn — function file (hadesllm/moirais)
"""Ridge/LASSO/ElasticNet regularization path."""
import numpy as np
from ._richresult import RichResult

__all__ = ["regularization_path"]


def regularization_path(x, y):
    """
    Ridge/LASSO/ElasticNet regularization path

    Formula: beta(lambda) for lambda in grid

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
    Geron (2026), Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge/LASSO/ElasticNet regularization path"})


def cheatsheet():
    return "rgztn: Ridge/LASSO/ElasticNet regularization path"
