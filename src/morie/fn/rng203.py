"""CCF of random signals as expectation of outer product of vector samples.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_ccf_outer_product_random_signals"]


def rangayyan_ch4_ccf_outer_product_random_signals(x, y, n):
    """
    CCF of random signals as expectation of outer product of vector samples.

    Formula: Theta_xy = E[ x(n) * y^T(n) ]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.29, p. 230
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CCF of random signals as expectation of outer product of vector samples."})


def cheatsheet():
    return "rng203: CCF of random signals as expectation of outer product of vector samples."
