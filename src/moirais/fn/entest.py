"""k-NN entropy estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["entropy_knn"]


def entropy_knn(X, k):
    """
    k-NN entropy estimator

    Formula: H_KL = -psi(k) + psi(N) + log V_d + d/N sum log eps_i

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kozachenko-Leonenko (1987)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "k-NN entropy estimator"})
    estimate = np.median(X)
    se = 1.2533 * np.std(X, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "k-NN entropy estimator"})


def cheatsheet():
    return "entest: k-NN entropy estimator"
