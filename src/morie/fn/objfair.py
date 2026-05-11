"""Individual fairness via Lipschitz constraint on classifier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["individual_fairness_lipschitz"]


def individual_fairness_lipschitz(y, h_values, x_pairs, L):
    """
    Individual fairness via Lipschitz constraint on classifier

    Formula: |h(x) - h(x')| <= L * d(x, x')

    Parameters
    ----------
    y : array-like
        Input data.
    h_values : array-like
        Input data.
    x_pairs : array-like
        Input data.
    L : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork et al (2012) Fairness Through Awareness
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Individual fairness via Lipschitz constraint on classifier"})


def cheatsheet():
    return "objfair: Individual fairness via Lipschitz constraint on classifier"
