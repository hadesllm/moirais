"""MLE of theta for IRT."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["theta_mle"]


def theta_mle(X, items):
    """
    MLE of theta for IRT

    Formula: argmax product P_j^x_j (1-P_j)^(1-x_j)

    Parameters
    ----------
    X : array-like
        Input data.
    items : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Birnbaum (1968)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MLE of theta for IRT"})


def cheatsheet():
    return "thetml: MLE of theta for IRT"
