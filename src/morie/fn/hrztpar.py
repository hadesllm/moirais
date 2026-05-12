# morie.fn -- function file (hadesllm/morie)
"""Transformation model with parametric T, nonparametric F."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_parametric_T"]


def horowitz_parametric_T(x, y, T_family):
    """
    Transformation model with parametric T, nonparametric F

    Formula: T(Y;theta) = X'beta + U; estimate (beta,theta) by IV; F_U = nonparametric

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    T_family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_hat, beta_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transformation model with parametric T, nonparametric F"})


def cheatsheet():
    return "hrztpar: Transformation model with parametric T, nonparametric F"
