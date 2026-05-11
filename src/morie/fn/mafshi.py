"""Inverse Fisher z back to correlation r."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_fishers_z_inverse"]


def ma_fishers_z_inverse(z):
    """
    Inverse Fisher z back to correlation r

    Formula: r = (e^{2z}-1)/(e^{2z}+1)

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: r

    References
    ----------
    Fisher (1921)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(z), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Inverse Fisher z back to correlation r"})
    result = stats.spearmanr(z[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Inverse Fisher z back to correlation r"})


def cheatsheet():
    return "mafshi: Inverse Fisher z back to correlation r"
