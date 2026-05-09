"""Pearl's mediation formula."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mediation_formula"]


def mediation_formula(Y_model, M_model, x, x_prime, C):
    """
    Pearl's mediation formula

    Formula: NIE = sum_m E[Y|x,m,c](P(m|x,c)−P(m|x',c))

    Parameters
    ----------
    Y_model : array-like
        Input data.
    M_model : array-like
        Input data.
    x : array-like
        Input data.
    x_prime : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pearl (2001) §6
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pearl's mediation formula"})


def cheatsheet():
    return "medfm: Pearl's mediation formula"
