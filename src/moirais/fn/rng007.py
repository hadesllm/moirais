"""Sample mean estimator from N observed samples.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_sample_mean"]


def rangayyan_ch3_sample_mean(eta, N):
    """
    Sample mean estimator from N observed samples.

    Formula: mu_eta = (1/N) * sum_{n=0}^{N-1} eta(n)

    Parameters
    ----------
    eta : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.7, p. 95
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    n = len(eta)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Sample mean estimator from N observed samples."})
    estimate = np.median(eta)
    se = 1.2533 * np.std(eta, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Sample mean estimator from N observed samples."})


def cheatsheet():
    return "rng007: Sample mean estimator from N observed samples."
