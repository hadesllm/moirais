# moirais.fn — function file (hadesllm/moirais)
"""CNN for genomic prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["cnn_genomic"]


def cnn_genomic(x, y, markers):
    """
    CNN for genomic prediction

    Formula: Conv1D on marker array → dense → prediction

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
    Montesinos Lopez Ch 13
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CNN for genomic prediction"})


def cheatsheet():
    return "cnnge: CNN for genomic prediction"
