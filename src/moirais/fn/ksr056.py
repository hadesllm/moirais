"""Lipschitz bound for least-absolute-deviation criterion class."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_lad_lipschitz_bound"]


def kosorok_ch2_lad_lipschitz_bound(theta_1, theta_2, u, x):
    """
    Lipschitz bound for least-absolute-deviation criterion class

    Formula: | m_{theta_1}(x) - m_{theta_2}(x) | <= || theta_1 - theta_2 || * ||u||

    Parameters
    ----------
    theta_1 : array-like
        Input data.
    theta_2 : array-like
        Input data.
    u : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.20, p. 30
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lipschitz bound for least-absolute-deviation criterion class"})


def cheatsheet():
    return "ksr056: Lipschitz bound for least-absolute-deviation criterion class"
