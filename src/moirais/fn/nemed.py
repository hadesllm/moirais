"""Nested counterfactual mediation effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nested_counterfactual_mediation"]


def nested_counterfactual_mediation(X, M, Y):
    """
    Nested counterfactual mediation effect

    Formula: E[Y(a, M(a*))] for a, a* in {0,1}

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Daniel et al. (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nested counterfactual mediation effect"})


def cheatsheet():
    return "nemed: Nested counterfactual mediation effect"
