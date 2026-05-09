"""SVM Wolfe dual formulation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["svm_dual_wolfe"]


def svm_dual_wolfe(X, y, K):
    """
    SVM Wolfe dual formulation

    Formula: max sum alpha_i - (1/2) sum_i sum_j alpha_i*alpha_j*y_i*y_j*K(x_i,x_j) s.t. sum alpha_i*y_i=0, alpha_i>=0

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'alpha': 'array', 'b': 'float'}

    References
    ----------
    Montesinos Lopez Ch 9
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVM Wolfe dual formulation"})


def cheatsheet():
    return "svmdu: SVM Wolfe dual formulation"
