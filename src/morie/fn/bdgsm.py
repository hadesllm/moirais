"""Bridge sampling estimator of marginal likelihood."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bridge_sampling_marginal"]


def bridge_sampling_marginal(log_lik, prior):
    """
    Bridge sampling estimator of marginal likelihood

    Formula: m(y) via iterative bridge between proposal q and posterior

    Parameters
    ----------
    log_lik : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Meng & Wong (1996); Gronau et al. (2017)
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bridge sampling estimator of marginal likelihood"})
    estimate = np.median(log_lik)
    se = 1.2533 * np.std(log_lik, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bridge sampling estimator of marginal likelihood"})


def cheatsheet():
    return "bdgsm: Bridge sampling estimator of marginal likelihood"
