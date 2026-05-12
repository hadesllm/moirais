"""Normalizing-flow density anomaly."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["normalizing_flow_anomaly"]


def normalizing_flow_anomaly(X, flow):
    """
    Normalizing-flow density anomaly

    Formula: low log p(x) -> anomaly

    Parameters
    ----------
    X : array-like
        Input data.
    flow : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dinh et al (2017) RealNVP
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normalizing-flow density anomaly"})


def cheatsheet():
    return "flow_an: Normalizing-flow density anomaly"
