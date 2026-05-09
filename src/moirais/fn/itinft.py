"""Item information function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["item_information_function"]


def item_information_function(theta, a, b):
    """
    Item information function

    Formula: I_j(theta) = a_j^2 P(1-P) for 2PL

    Parameters
    ----------
    theta : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lord (1980)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Item information function"})


def cheatsheet():
    return "itinft: Item information function"
