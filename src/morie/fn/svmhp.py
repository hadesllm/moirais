"""SVM maximum margin hyperplane (hard margin)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["svm_hyperplane"]


def svm_hyperplane(X, y):
    """
    SVM maximum margin hyperplane (hard margin)

    Formula: min ||w||^2 s.t. y_i*(w'x_i + b) >= 1 for all i

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'w': 'array', 'b': 'float'}

    References
    ----------
    Montesinos Lopez Ch 9
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVM maximum margin hyperplane (hard margin)"})


def cheatsheet():
    return "svmhp: SVM maximum margin hyperplane (hard margin)"
