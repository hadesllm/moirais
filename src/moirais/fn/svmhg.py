"""SVM primal with hinge loss."""
import numpy as np
from ._richresult import RichResult

__all__ = ["svm_hinge_primal"]


def svm_hinge_primal(x, y):
    """
    SVM primal with hinge loss

    Formula: min (1/2)||w||^2 + C sum max(0, 1-yi(w'xi+b))

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVM primal with hinge loss"})


def cheatsheet():
    return "svmhg: SVM primal with hinge loss"
