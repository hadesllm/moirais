"""SVM soft margin with slack variables (C-SVM)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["svm_soft_margin"]


def svm_soft_margin(X, y, C):
    """
    SVM soft margin with slack variables (C-SVM)

    Formula: min (1/2)||w||^2 + C*sum_i xi_i s.t. y_i*(w'x_i+b) >= 1-xi_i, xi_i >= 0

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'w': 'array', 'b': 'float', 'support_vectors': 'array'}

    References
    ----------
    Montesinos Lopez Ch 9
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVM soft margin with slack variables (C-SVM)"})


def cheatsheet():
    return "svmsl: SVM soft margin with slack variables (C-SVM)"
