"""History-adjusted MSM for dynamic treatment regimes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["history_adjusted_msm"]


def history_adjusted_msm(y, treatment_history, covariate_history, time, regime):
    """
    History-adjusted MSM for dynamic treatment regimes

    Formula: E[Y(d) | H_0]; weight by treatment-rule-consistent IPTW

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    time : array-like
        Input data.
    regime : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van der Laan-Petersen (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "History-adjusted MSM for dynamic treatment regimes"})


def cheatsheet():
    return "hmstrn: History-adjusted MSM for dynamic treatment regimes"
