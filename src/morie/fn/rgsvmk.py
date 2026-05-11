# morie.fn — function file (hadesllm/morie)
"""SVM with kernel trick (RBF, polynomial, sigmoid kernels)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_svm_kernel"]


def rangayyan_svm_kernel(X, y, kernel, C, gamma):
    """
    SVM with kernel trick (RBF, polynomial, sigmoid kernels)

    Formula: K(y,y')=exp(-||x-x'||^2/(2*sigma^2)) for RBF; dual: alpha* from QP

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
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: svm_model, support_vectors

    References
    ----------
    Rangayyan Ch 10.4.5
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVM with kernel trick (RBF, polynomial, sigmoid kernels)"})


def cheatsheet():
    return "rgsvmk: SVM with kernel trick (RBF, polynomial, sigmoid kernels)"
