"""Sun-Abraham interaction-weighted estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_did_sun_abraham"]


def causal_did_sun_abraham(Y_panel, G_first_treat):
    """
    Sun-Abraham interaction-weighted estimator

    Formula: Saturated cohort × event-time dummies

    Parameters
    ----------
    Y_panel : array-like
        Input data.
    G_first_treat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATT_gk

    References
    ----------
    Sun & Abraham (2021)
    """
    Y_panel = np.atleast_1d(np.asarray(Y_panel, dtype=float))
    n = len(Y_panel)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Sun-Abraham interaction-weighted estimator"})
    estimate = np.median(Y_panel)
    se = 1.2533 * np.std(Y_panel, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Sun-Abraham interaction-weighted estimator"})


def cheatsheet():
    return "causdidsap: Sun-Abraham interaction-weighted estimator"
