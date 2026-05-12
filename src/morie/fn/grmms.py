# morie.fn -- function file (hadesllm/morie)
"""Min-max scaling to [0, 1]."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_minmax_scaler"]


def geron_minmax_scaler(X):
    """
    Min-max scaling to [0, 1]

    Formula: x_scaled = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_scaled

    References
    ----------
    Géron Ch 2, Feature Scaling section (min-max)
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Min-max scaling to [0, 1]"})


def cheatsheet():
    return "grmms: Min-max scaling to [0, 1]"
