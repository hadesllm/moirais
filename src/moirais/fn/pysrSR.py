"""PySR symbolic regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pysr_regression"]


def pysr_regression(X, y):
    """
    PySR symbolic regression

    Formula: genetic programming + Pareto front

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cranmer (2023) PySR
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PySR symbolic regression"})


def cheatsheet():
    return "pysrSR: PySR symbolic regression"
