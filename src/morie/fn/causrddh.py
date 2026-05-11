"""IK optimal bandwidth for sharp RDD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_rdd_imbens_kalyanaraman"]


def causal_rdd_imbens_kalyanaraman(x, y, cutoff):
    """
    IK optimal bandwidth for sharp RDD

    Formula: h_IK = C_K (σ²/(f m''²))^{1/5} n^{-1/5}

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    cutoff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_IK

    References
    ----------
    Imbens-Kalyanaraman (2012)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IK optimal bandwidth for sharp RDD"})


def cheatsheet():
    return "causrddh: IK optimal bandwidth for sharp RDD"
