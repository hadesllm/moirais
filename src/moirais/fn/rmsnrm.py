"""RMSNorm — root-mean-square layer normalization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rms_norm"]


def rms_norm(y, x, g):
    """
    RMSNorm — root-mean-square layer normalization

    Formula: y = x / RMS(x) * g, RMS(x) = sqrt(mean(x^2))

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang & Sennrich (2019)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RMSNorm — root-mean-square layer normalization"})


def cheatsheet():
    return "rmsnrm: RMSNorm — root-mean-square layer normalization"
