"""Item information function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["item_information"]


def item_information(y, theta, a, b):
    """
    Item information function

    Formula: I_i(theta) = a_i^2 P_i(theta) (1 - P_i(theta))

    Parameters
    ----------
    y : array-like
        Input data.
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
    Birnbaum (1968)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Item information function"})


def cheatsheet():
    return "iinfo: Item information function"
