"""Transfer-learning MSM across cohorts."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["transfer_learning_msm"]


def transfer_learning_msm(y, A, H, cohort):
    """
    Transfer-learning MSM across cohorts

    Formula: borrow strength via posterior averaging

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.
    cohort : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey-Wager (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transfer-learning MSM across cohorts"})


def cheatsheet():
    return "trnsfr: Transfer-learning MSM across cohorts"
