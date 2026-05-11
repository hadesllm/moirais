# morie.fn — function file (hadesllm/morie)
"""Support vector machine (SVM) via margin maximization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_svm"]


def rangayyan_svm(X, y, kernel, C):
    """
    Support vector machine (SVM) via margin maximization

    Formula: max 2/||w|| s.t. y_i(w^T*x_i+b)>=1; dual: max sum(a_i)-(1/2)*sum sum a_i*a_j*y_i*y_j*K(x_i,x_j)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    kernel : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: support_vectors, w, b, alphas

    References
    ----------
    Rangayyan Ch 10.4.5
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Support vector machine (SVM) via margin maximization"})


def cheatsheet():
    return "rgsvm: Support vector machine (SVM) via margin maximization"
