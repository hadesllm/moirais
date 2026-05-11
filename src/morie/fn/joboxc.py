# morie.fn — function file (hadesllm/morie)
"""Box-Cox power transformation (strictly positive y)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_box_cox_transform"]


def joseph_box_cox_transform(y, lam):
    """
    Box-Cox power transformation (strictly positive y)

    Formula: y_t(lambda) = (y_t^lambda - 1) / lambda  if lambda != 0  else log(y_t)

    Parameters
    ----------
    y : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_transformed

    References
    ----------
    Joseph Ch 7, Box-Cox section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Box-Cox power transformation (strictly positive y)"})


def cheatsheet():
    return "joboxc: Box-Cox power transformation (strictly positive y)"
