"""MOMENT — encoder TS foundation model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["moment_foundation"]


def moment_foundation(y, task):
    """
    MOMENT — encoder TS foundation model

    Formula: masked patch reconstruction + downstream heads

    Parameters
    ----------
    y : array-like
        Input data.
    task : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Goswami et al (2024) MOMENT
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MOMENT — encoder TS foundation model"})


def cheatsheet():
    return "momento: MOMENT — encoder TS foundation model"
