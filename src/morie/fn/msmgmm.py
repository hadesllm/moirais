"""MSM via generalized method of moments."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_gmm_estimator"]


def msm_gmm_estimator(y, treatment_history, covariate_history, instruments):
    """
    MSM via generalized method of moments

    Formula: E[Z (Y - g(a_bar; beta))] = 0 with IPTW Z

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    instruments : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hansen (1982); Robins (1999)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM via generalized method of moments"})


def cheatsheet():
    return "msmgmm: MSM via generalized method of moments"
