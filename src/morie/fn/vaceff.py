"""Vaccine efficacy estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vaccine_efficacy"]


def vaccine_efficacy(incidence_v, incidence_u, person_time):
    """
    Vaccine efficacy estimate

    Formula: VE = 1 - IRR_vaccinated_vs_unvaccinated

    Parameters
    ----------
    incidence_v : array-like
        Input data.
    incidence_u : array-like
        Input data.
    person_time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Halloran et al (2010)
    """
    incidence_v = np.atleast_1d(np.asarray(incidence_v, dtype=float))
    n = len(incidence_v)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Vaccine efficacy estimate"})
    estimate = np.median(incidence_v)
    se = 1.2533 * np.std(incidence_v, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Vaccine efficacy estimate"})


def cheatsheet():
    return "vaceff: Vaccine efficacy estimate"
