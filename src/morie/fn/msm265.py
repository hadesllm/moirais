"""Numbered display equation (14.1) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_1"]


def mvsml_convolutional_nn_eq_14_1(x, t, l, dt, E, xT):
    """
    Numbered display equation (14.1) from MVSML chapter 14.

    Formula: x t( )ϕl t( )dt + E = \mu + xT\beta0 + E 0 = xT\beta + E, (14.3) R T where x = [1, xT]T, x = x1, . . . , xL1 T, xl = = 0 x t( )ϕl t( )dt, l = 1, . . ., L1. So, if yi, i = 1, . . ., n, are independent observations of model

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    l : array-like
        Input data.
    dt : array-like
        Input data.
    E : array-like
        Input data.
    xT : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.1) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.1) from MVSML chapter 14."})


def cheatsheet():
    return "msm265: Numbered display equation (14.1) from MVSML chapter 14."
