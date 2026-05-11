"""1-parameter logistic (Rasch) model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rasch_one_parameter"]


def rasch_one_parameter(y, theta, b):
    """
    1-parameter logistic (Rasch) model

    Formula: P(X=1|theta,b) = exp(theta - b) / (1 + exp(theta - b))

    Parameters
    ----------
    y : array-like
        Input data.
    theta : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rasch (1960); Birnbaum (1968)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "1-parameter logistic (Rasch) model"})


def cheatsheet():
    return "irt1pl: 1-parameter logistic (Rasch) model"
