"""Re-identification risk estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["reidentification_risk"]


def reidentification_risk(sample, population, quasi_ids):
    """
    Re-identification risk estimate

    Formula: P(unique on QIDs) over population

    Parameters
    ----------
    sample : array-like
        Input data.
    population : array-like
        Input data.
    quasi_ids : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    El Emam et al (2011)
    """
    sample = np.atleast_1d(np.asarray(sample, dtype=float))
    n = len(sample)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Re-identification risk estimate"})
    estimate = np.median(sample)
    se = 1.2533 * np.std(sample, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Re-identification risk estimate"})


def cheatsheet():
    return "reidR: Re-identification risk estimate"
