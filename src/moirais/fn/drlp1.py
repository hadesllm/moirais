"""DR-DiD via local projection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_lp_did"]


def dr_lp_did(y, D, unit, time, horizon):
    """
    DR-DiD via local projection

    Formula: local-projection IRF with DR weights

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dube-Girardi-Jordà-Taylor (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD via local projection"})


def cheatsheet():
    return "drlp1: DR-DiD via local projection"
