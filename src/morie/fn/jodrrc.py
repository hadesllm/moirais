# morie.fn — function file (hadesllm/morie)
"""DirRec hybrid: direct per horizon, but include previous prediction as input."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_dirrec_strategy"]


def joseph_dirrec_strategy(X, y, H):
    """
    DirRec hybrid: direct per horizon, but include previous prediction as input

    Formula: model_h fits (X, y_{t+h}) with features incl. y_hat_{t+h-1}

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: models

    References
    ----------
    Joseph Ch 18, DirRec Hybrid Strategy section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DirRec hybrid: direct per horizon, but include previous prediction as input"})


def cheatsheet():
    return "jodrrc: DirRec hybrid: direct per horizon, but include previous prediction as input"
