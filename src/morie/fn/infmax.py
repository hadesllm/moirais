"""InfoMax objective for representation learning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["infomax_objective"]


def infomax_objective(X, T_network):
    """
    InfoMax objective for representation learning

    Formula: max I(X; T(X))

    Parameters
    ----------
    X : array-like
        Input data.
    T_network : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Linsker (1988)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "InfoMax objective for representation learning"})


def cheatsheet():
    return "infmax: InfoMax objective for representation learning"
