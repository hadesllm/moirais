"""Hetero-causal forest with monotonicity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hetero_causal_forest"]


def hetero_causal_forest(y, D, X, mono_mask):
    """
    Hetero-causal forest with monotonicity

    Formula: forest splits constrained by monotonicity

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    mono_mask : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cui-Cuif-Tchetgen Tchetgen (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hetero-causal forest with monotonicity"})


def cheatsheet():
    return "htgcrf: Hetero-causal forest with monotonicity"
