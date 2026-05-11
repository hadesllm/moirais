"""Ensemble averaged function of time as a prototype of M observations.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ensemble_average_function"]


def rangayyan_ch3_ensemble_average_function(x_k, M):
    """
    Ensemble averaged function of time as a prototype of M observations.

    Formula: x_bar(t) = mu_x(t) = (1/M) * sum_{k=1}^{M} x_k(t)

    Parameters
    ----------
    x_k : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.18, p. 97
    """
    x_k = np.atleast_1d(np.asarray(x_k, dtype=float))
    n = len(x_k)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Ensemble averaged function of time as a prototype of M observations."})
    estimate = np.median(x_k)
    se = 1.2533 * np.std(x_k, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Ensemble averaged function of time as a prototype of M observations."})


def cheatsheet():
    return "rng018: Ensemble averaged function of time as a prototype of M observations."
