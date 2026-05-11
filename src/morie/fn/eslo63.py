""".632 estimator combining train and bootstrap."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_oob_632"]


def esl_oob_632(err_train, err_boot):
    """
    .632 estimator combining train and bootstrap

    Formula: Err_.632 = .368 train + .632 boot

    Parameters
    ----------
    err_train : array-like
        Input data.
    err_boot : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 7
    """
    err_train = np.atleast_1d(np.asarray(err_train, dtype=float))
    n = len(err_train)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": ".632 estimator combining train and bootstrap"})
    estimate = np.median(err_train)
    se = 1.2533 * np.std(err_train, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": ".632 estimator combining train and bootstrap"})


def cheatsheet():
    return "eslo63: .632 estimator combining train and bootstrap"
