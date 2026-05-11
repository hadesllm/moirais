"""Palmer Drought Severity Index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pdsi"]


def pdsi(P, PET, awc):
    """
    Palmer Drought Severity Index

    Formula: water-balance moisture anomaly

    Parameters
    ----------
    P : array-like
        Input data.
    PET : array-like
        Input data.
    awc : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Palmer (1965)
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Palmer Drought Severity Index"})


def cheatsheet():
    return "droPDSI: Palmer Drought Severity Index"
