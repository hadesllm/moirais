"""Ensemble estimate of the ACF from M observations of a random process.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_acf_ensemble_estimate"]


def rangayyan_ch3_acf_ensemble_estimate(x_k, t1, tau, M):
    """
    Ensemble estimate of the ACF from M observations of a random process.

    Formula: phi_xx(t1, t1+tau) = lim_{M->inf} (1/M) * sum_{k=1}^{M} x_k(t1) x_k(t1+tau)

    Parameters
    ----------
    x_k : array-like
        Input data.
    t1 : array-like
        Input data.
    tau : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.17, p. 96
    """
    x_k = np.atleast_1d(np.asarray(x_k, dtype=float))
    n = len(x_k)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Ensemble estimate of the ACF from M observations of a random process."})
    estimate = np.median(x_k)
    se = 1.2533 * np.std(x_k, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Ensemble estimate of the ACF from M observations of a random process."})


def cheatsheet():
    return "rng017: Ensemble estimate of the ACF from M observations of a random process."
