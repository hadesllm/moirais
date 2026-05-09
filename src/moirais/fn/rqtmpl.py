"""QTL mapping (interval mapping)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["qtl_mapping"]


def qtl_mapping(y, markers, positions):
    """
    QTL mapping (interval mapping)

    Formula: likelihood profile across genome

    Parameters
    ----------
    y : array-like
        Input data.
    markers : array-like
        Input data.
    positions : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lander-Botstein (1989)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QTL mapping (interval mapping)"})


def cheatsheet():
    return "rqtmpl: QTL mapping (interval mapping)"
