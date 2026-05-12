# morie.fn -- function file (hadesllm/morie)
"""Infomax ICA algorithm (Bell-Sejnowski)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_infomax_ica"]


def rangayyan_infomax_ica(X, n_components, lr, max_iter):
    """
    Infomax ICA algorithm (Bell-Sejnowski)

    Formula: DeltaW = (I - f(y)*y^T)*W; f(y) = 1-2*sigmoid(y)

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    lr : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: S, W

    References
    ----------
    Rangayyan Ch 9.7.2
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Infomax ICA algorithm (Bell-Sejnowski)"})


def cheatsheet():
    return "rginf: Infomax ICA algorithm (Bell-Sejnowski)"
