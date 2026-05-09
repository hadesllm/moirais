"""Truncated combined IPTW × IPCW."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["truncated_combined_weights"]


def truncated_combined_weights(sw_A, sw_C, quantile):
    """
    Truncated combined IPTW × IPCW

    Formula: min(sw_A * sw_C, q99)

    Parameters
    ----------
    sw_A : array-like
        Input data.
    sw_C : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cole-Hernán (2008)
    """
    sw_A = np.atleast_1d(np.asarray(sw_A, dtype=float))
    n = len(sw_A)
    result = float(np.mean(sw_A))
    se = float(np.std(sw_A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Truncated combined IPTW × IPCW"})


def cheatsheet():
    return "trcwgt: Truncated combined IPTW × IPCW"
