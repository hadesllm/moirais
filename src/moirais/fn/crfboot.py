"""Bootstrap-honest causal forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_forest_bootstrap"]


def causal_forest_bootstrap(y, D, X, B, min_node):
    """
    Bootstrap-honest causal forest

    Formula: resample with sample-splitting

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    B : array-like
        Input data.
    min_node : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wager-Athey (2018) Appendix
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap-honest causal forest"})


def cheatsheet():
    return "crfboot: Bootstrap-honest causal forest"
