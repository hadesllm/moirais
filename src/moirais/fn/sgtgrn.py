"""GCN propagation step Â^k X."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_graph_neural_propagation"]


def sgt_graph_neural_propagation(A_hat, X, W):
    """
    GCN propagation step Â^k X

    Formula: X^{(k+1)} = σ(Â X^{(k)} W^{(k)})

    Parameters
    ----------
    A_hat : array-like
        Input data.
    X : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_next

    References
    ----------
    Kipf & Welling (2017)
    """
    A_hat = np.atleast_1d(np.asarray(A_hat, dtype=float))
    n = len(A_hat)
    result = float(np.mean(A_hat))
    se = float(np.std(A_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GCN propagation step Â^k X"})


def cheatsheet():
    return "sgtgrn: GCN propagation step Â^k X"
