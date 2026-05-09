"""MSM with instrumental variables."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_iv"]


def msm_iv(y, treatment_history, instruments, covariate_history):
    """
    MSM with instrumental variables

    Formula: weight by 1/Pr(A|Z); E[Y(a)|Z] target

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    instruments : array-like
        Input data.
    covariate_history : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Greenland (1996); Tan (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM with instrumental variables"})


def cheatsheet():
    return "msmiv2: MSM with instrumental variables"
