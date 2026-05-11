# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Binary cross-entropy loss for binary genomic outcomes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["binary_crossentropy_loss"]


def binary_crossentropy_loss(y, p):
    """
    Binary cross-entropy loss for binary genomic outcomes

    Formula: L = -(1/n) * sum_i [y_i*log(p_i) + (1-y_i)*log(1-p_i)]

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'loss': 'float'}

    References
    ----------
    Montesinos Lopez Ch 10
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Binary cross-entropy loss for binary genomic outcomes"})


def cheatsheet():
    return "bcelO: Binary cross-entropy loss for binary genomic outcomes"
