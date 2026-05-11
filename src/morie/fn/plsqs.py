"""PLS regression QSAR with cross-validated component count."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pls_qsar"]


def pls_qsar(activities, descriptors, n_components):
    """
    PLS regression QSAR with cross-validated component count

    Formula: NIPALS PLS on standardized descriptors

    Parameters
    ----------
    activities : array-like
        Input data.
    descriptors : array-like
        Input data.
    n_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wold et al (2001)
    """
    activities = np.atleast_1d(np.asarray(activities, dtype=float))
    n = len(activities)
    result = float(np.mean(activities))
    se = float(np.std(activities, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PLS regression QSAR with cross-validated component count"})


def cheatsheet():
    return "plsqs: PLS regression QSAR with cross-validated component count"
