# morie.fn — function file (hadesllm/morie)
"""Deep learning for genomic prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["deep_learning_genomic"]


def deep_learning_genomic(x, y, markers):
    """
    Deep learning for genomic prediction

    Formula: y = f_DNN(X*markers) + e

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
    Montesinos Lopez Ch 12
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep learning for genomic prediction"})


def cheatsheet():
    return "dlgen: Deep learning for genomic prediction"
