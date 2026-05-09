"""Central limit theorem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_clt"]


def wasserman_clt(data):
    """
    Central limit theorem

    Formula: sqrt(n)(X_bar - mu)/sigma ~> N(0,1)

    Parameters
    ----------
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 5
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Central limit theorem"})


def cheatsheet():
    return "wsmclt: Central limit theorem"
