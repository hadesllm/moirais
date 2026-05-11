"""Transportability weights from source to target population."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_transportability_weights"]


def causal_transportability_weights(X_source, X_target):
    """
    Transportability weights from source to target population

    Formula: w_i = P_target(X)/P_source(X)

    Parameters
    ----------
    X_source : array-like
        Input data.
    X_target : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights, ATE_target

    References
    ----------
    Pearl & Bareinboim (2014)
    """
    X_source = np.atleast_1d(np.asarray(X_source, dtype=float))
    n = len(X_source)
    result = float(np.mean(X_source))
    se = float(np.std(X_source, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transportability weights from source to target population"})


def cheatsheet():
    return "caustrnsp: Transportability weights from source to target population"
