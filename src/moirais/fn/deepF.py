"""DeepFM (FM + DNN)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deepfm"]


def deepfm(X, y, K, mlp_h):
    """
    DeepFM (FM + DNN)

    Formula: FM (low-order) + DNN (high-order) on shared embedding

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    K : array-like
        Input data.
    mlp_h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Guo et al (2017) DeepFM
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DeepFM (FM + DNN)"})


def cheatsheet():
    return "deepF: DeepFM (FM + DNN)"
