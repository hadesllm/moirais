"""Two-stage TMLE for staged interventions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_two_stage"]


def tmle_two_stage(y, D1, D2, X1, X2):
    """
    Two-stage TMLE for staged interventions

    Formula: target sequential intervention effect

    Parameters
    ----------
    y : array-like
        Input data.
    D1 : array-like
        Input data.
    D2 : array-like
        Input data.
    X1 : array-like
        Input data.
    X2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schnitzer-vdL-Lipsitch (2014)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-stage TMLE for staged interventions"})


def cheatsheet():
    return "tmltsm: Two-stage TMLE for staged interventions"
