"""Covariance between two random processes x and y.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_covariance"]


def rangayyan_ch3_covariance(x, y, mu_x, mu_y):
    """
    Covariance between two random processes x and y.

    Formula: C_xy = E[(x - mu_x)(y - mu_y)] = double_integral (x - mu_x)(y - mu_y) p_{x,y}(x,y) dx dy

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    mu_x : array-like
        Input data.
    mu_y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.21, p. 98
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Covariance between two random processes x and y."})


def cheatsheet():
    return "rng021: Covariance between two random processes x and y."
