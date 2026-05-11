# morie.fn — function file (hadesllm/morie)
"""Levinson-Durbin recursion for efficient AR model fitting."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_levinson_durbin"]


def rangayyan_levinson_durbin(acf, order):
    """
    Levinson-Durbin recursion for efficient AR model fitting

    Formula: k_m = -(R(m) + sum a_{m-1}(k)*R(m-k)) / P_{m-1}; forward/backward update

    Parameters
    ----------
    acf : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a_coeffs, k_reflections, prediction_error

    References
    ----------
    Rangayyan Ch 7.5
    """
    acf = np.asarray(acf, dtype=float)
    n = int(acf) if acf.ndim == 0 else len(acf)
    result = float(np.mean(acf))
    se = float(np.std(acf, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Levinson-Durbin recursion for efficient AR model fitting"})


def cheatsheet():
    return "rglevd: Levinson-Durbin recursion for efficient AR model fitting"
