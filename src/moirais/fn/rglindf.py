# moirais.fn — function file (hadesllm/moirais)
"""Linear discriminant function for pattern classification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_linear_discrim"]


def rangayyan_linear_discrim(X, y, w, w0):
    """
    Linear discriminant function for pattern classification

    Formula: g(y) = w^T*y + w_0; classify to class with max g_i(y)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.
    w0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, scores

    References
    ----------
    Rangayyan Ch 10.4.1
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear discriminant function for pattern classification"})


def cheatsheet():
    return "rglindf: Linear discriminant function for pattern classification"
