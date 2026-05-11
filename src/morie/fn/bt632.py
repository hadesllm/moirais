""".632 error estimator combining apparent and OOB."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_632_estimator"]


def boot_632_estimator(err_app, err_oob):
    """
    .632 error estimator combining apparent and OOB

    Formula: ê_.632 = .368 ê_app + .632 ê_OOB

    Parameters
    ----------
    err_app : array-like
        Input data.
    err_oob : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: err_632

    References
    ----------
    Efron & Tibshirani (1997)
    """
    err_app = np.atleast_1d(np.asarray(err_app, dtype=float))
    n = len(err_app)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": ".632 error estimator combining apparent and OOB"})
    estimate = np.median(err_app)
    se = 1.2533 * np.std(err_app, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": ".632 error estimator combining apparent and OOB"})


def cheatsheet():
    return "bt632: .632 error estimator combining apparent and OOB"
