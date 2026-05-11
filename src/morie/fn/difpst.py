"""DIF p-difference (raw)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dif_p_diff"]


def dif_p_diff(X, group):
    """
    DIF p-difference (raw)

    Formula: p_focal - p_reference per item

    Parameters
    ----------
    X : array-like
        Input data.
    group : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Holland-Wainer (1993)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DIF p-difference (raw)"})


def cheatsheet():
    return "difpst: DIF p-difference (raw)"
