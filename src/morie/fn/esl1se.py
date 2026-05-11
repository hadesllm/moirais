"""One-standard-error rule for tuning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_one_se_rule"]


def esl_one_se_rule(cv_err, cv_se):
    """
    One-standard-error rule for tuning

    Formula: Pick simplest model within 1 SE of min CV error

    Parameters
    ----------
    cv_err : array-like
        Input data.
    cv_se : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lambda_chosen

    References
    ----------
    Hastie ESL Ch 7
    """
    cv_err = np.atleast_1d(np.asarray(cv_err, dtype=float))
    n = len(cv_err)
    result = float(np.mean(cv_err))
    se = float(np.std(cv_err, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "One-standard-error rule for tuning"})


def cheatsheet():
    return "esl1se: One-standard-error rule for tuning"
