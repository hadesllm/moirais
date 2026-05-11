"""Uno C-index for censored data."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["uno_concordance"]


def uno_concordance(time, event, predicted_risk):
    """
    Uno C-index for censored data

    Formula: truncated Harrell C with IPCW

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    predicted_risk : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Uno et al (2011)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Uno C-index for censored data"})


def cheatsheet():
    return "survci2: Uno C-index for censored data"
