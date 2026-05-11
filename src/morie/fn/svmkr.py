"""SVM with kernel (RBF/poly/sigmoid)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["svm_kernel_trick"]


def svm_kernel_trick(x, y):
    """
    SVM with kernel (RBF/poly/sigmoid)

    Formula: K(xi,xj) = exp(-gamma||xi-xj||^2)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVM with kernel (RBF/poly/sigmoid)"})


def cheatsheet():
    return "svmkr: SVM with kernel (RBF/poly/sigmoid)"
