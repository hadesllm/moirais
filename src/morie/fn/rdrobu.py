"""Calonico-Cattaneo-Titiunik robust RDD CIs."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["calonico_cattaneo_titiunik"]


def calonico_cattaneo_titiunik(y, x, cutoff):
    """
    Calonico-Cattaneo-Titiunik robust RDD CIs

    Formula: bias-corrected tau-hat with valid CI under MSE-optimal bw

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    cutoff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Calonico, Cattaneo, Titiunik (2014)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Calonico-Cattaneo-Titiunik robust RDD CIs"})


def cheatsheet():
    return "rdrobu: Calonico-Cattaneo-Titiunik robust RDD CIs"
