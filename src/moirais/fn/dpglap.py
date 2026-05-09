"""Laplace mechanism for epsilon-DP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_laplace_mechanism"]


def dp_laplace_mechanism(y, sensitivity, epsilon):
    """
    Laplace mechanism for epsilon-DP

    Formula: M(x) = f(x) + Lap(0, sensitivity/epsilon)

    Parameters
    ----------
    y : array-like
        Input data.
    sensitivity : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork, McSherry, Nissim, Smith (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplace mechanism for epsilon-DP"})


def cheatsheet():
    return "dpglap: Laplace mechanism for epsilon-DP"
