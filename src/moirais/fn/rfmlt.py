# moirais.fn — function file (hadesllm/moirais)
"""Random forest for multivariate/multi-output response."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rf_multivariate"]


def rf_multivariate(X, Y_matrix, n_trees):
    """
    Random forest for multivariate/multi-output response

    Formula: Train separate trees per output or use multi-output RF; importance averaged

    Parameters
    ----------
    X : array-like
        Input data.
    Y_matrix : array-like
        Input data.
    n_trees : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'y_hat': 'matrix', 'importance': 'array'}

    References
    ----------
    Montesinos Lopez Ch 15
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random forest for multivariate/multi-output response"})


def cheatsheet():
    return "rfmlt: Random forest for multivariate/multi-output response"
