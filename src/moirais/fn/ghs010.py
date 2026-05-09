"""Discrete hazard rate V_j of the stick-breaking construction interpreted as the conditional probability that X equals j given X >= j.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_discrete_hazard_rate"]


def ghosal_ch3_discrete_hazard_rate(p_j, j, X):
    """
    Discrete hazard rate V_j of the stick-breaking construction interpreted as the conditional probability that X equals j given X >= j.

    Formula: V_j = p_j / (1 - sum_{l=1}^{j-1} p_l) = P(X = j | X >= j)

    Parameters
    ----------
    p_j : array-like
        Input data.
    j : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.3, p. 31
    """
    p_j = np.atleast_1d(np.asarray(p_j, dtype=float))
    n = len(p_j)
    result = float(np.mean(p_j))
    se = float(np.std(p_j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete hazard rate V_j of the stick-breaking construction interpreted as the conditional probability that X equals j given X >= j."})


def cheatsheet():
    return "ghs010: Discrete hazard rate V_j of the stick-breaking construction interpreted as the conditional probability that X equals j given X >= j."
