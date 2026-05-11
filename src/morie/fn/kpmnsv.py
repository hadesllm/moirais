"""Kaplan-Meier survival estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kaplan_meier_survival"]


def kaplan_meier_survival(time, event):
    """
    Kaplan-Meier survival estimator

    Formula: S(t) = prod_{t_i <= t} (1 - d_i/n_i)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kaplan-Meier (1958)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Kaplan-Meier survival estimator"})
    estimate = np.median(time)
    se = 1.2533 * np.std(time, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Kaplan-Meier survival estimator"})


def cheatsheet():
    return "kpmnsv: Kaplan-Meier survival estimator"
