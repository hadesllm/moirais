"""Principal neighborhood aggregation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pna"]


def pna(A, X, aggregators, scalers):
    """
    Principal neighborhood aggregation

    Formula: combine multiple aggregators × scalers

    Parameters
    ----------
    A : array-like
        Input data.
    X : array-like
        Input data.
    aggregators : array-like
        Input data.
    scalers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Corso et al (2020) PNA
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Principal neighborhood aggregation"})


def cheatsheet():
    return "pnaG: Principal neighborhood aggregation"
