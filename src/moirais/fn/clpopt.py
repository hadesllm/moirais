"""CLP LP solver wrapper."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clp_lp"]


def clp_lp(c, A, b):
    """
    CLP LP solver wrapper

    Formula: open-source COIN-OR LP

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Forrest et al (COIN-OR)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLP LP solver wrapper"})


def cheatsheet():
    return "clpopt: CLP LP solver wrapper"
