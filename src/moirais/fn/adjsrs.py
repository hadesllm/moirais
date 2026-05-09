"""Effective SRS conversion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_srs"]


def effective_srs(design, method):
    """
    Effective SRS conversion

    Formula: convert complex design to equivalent SRS n_eff

    Parameters
    ----------
    design : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kish (1965)
    """
    design = np.atleast_1d(np.asarray(design, dtype=float))
    n = len(design)
    result = float(np.mean(design))
    se = float(np.std(design, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effective SRS conversion"})


def cheatsheet():
    return "adjsrs: Effective SRS conversion"
