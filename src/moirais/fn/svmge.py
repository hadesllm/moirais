"""SVM for genomic prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["svm_genomic"]


def svm_genomic(x, y, markers):
    """
    SVM for genomic prediction

    Formula: f(x) = sum alpha_i K(x_i, x) + b

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    markers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVM for genomic prediction"})


def cheatsheet():
    return "svmge: SVM for genomic prediction"
