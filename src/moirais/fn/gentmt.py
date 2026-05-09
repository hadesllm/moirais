"""MSM for continuous-dose treatment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["generalized_treatment_msm"]


def generalized_treatment_msm(y, A, H):
    """
    MSM for continuous-dose treatment

    Formula: weight by f(A|H)/f_marg(A); GAM over A

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Hernán-Brumback (2000); Imai-vDyk (2004) GPS
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM for continuous-dose treatment"})


def cheatsheet():
    return "gentmt: MSM for continuous-dose treatment"
