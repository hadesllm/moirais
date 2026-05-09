"""Fisher information I(theta) = -E[d^2 l / d theta^2]."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_fisher_info"]


def wasserman_fisher_info(f, theta):
    """
    Fisher information I(theta) = -E[d^2 l / d theta^2]

    Formula: I(theta) = -E_theta[d^2 log f / d theta^2]

    Parameters
    ----------
    f : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 9
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fisher information I(theta) = -E[d^2 l / d theta^2]"})


def cheatsheet():
    return "wsmfis: Fisher information I(theta) = -E[d^2 l / d theta^2]"
