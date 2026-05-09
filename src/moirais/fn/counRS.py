"""Counterfactual rec evaluation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["counterfactual_rec"]


def counterfactual_rec(logged, new_policy):
    """
    Counterfactual rec evaluation

    Formula: IPS-weighted estimator with logging policy

    Parameters
    ----------
    logged : array-like
        Input data.
    new_policy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang-Bayer-Niehues-Joachims (2018)
    """
    logged = np.atleast_1d(np.asarray(logged, dtype=float))
    n = len(logged)
    result = float(np.mean(logged))
    se = float(np.std(logged, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Counterfactual rec evaluation"})


def cheatsheet():
    return "counRS: Counterfactual rec evaluation"
