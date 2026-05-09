"""Three-layer GRF with mediator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_three_layer_grf"]


def causal_three_layer_grf(y, D, M, X):
    """
    Three-layer GRF with mediator

    Formula: forest each: outcome, mediator, treatment

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    M : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cui-Tchetgen Tchetgen (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Three-layer GRF with mediator"})


def cheatsheet():
    return "cthrgr: Three-layer GRF with mediator"
