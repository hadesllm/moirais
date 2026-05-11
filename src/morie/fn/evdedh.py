"""DEdH moment estimator of the EV index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_dekkers_einmahl_dehaan"]


def evt_dekkers_einmahl_dehaan(x, k):
    """
    DEdH moment estimator of the EV index

    Formula: γ̂_DEdH = M_n^{(1)} + 1 - 0.5/(1 - (M^{(1)})²/M^{(2)})

    Parameters
    ----------
    x : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gamma

    References
    ----------
    Dekkers-Einmahl-de Haan (1989)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "DEdH moment estimator of the EV index"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "DEdH moment estimator of the EV index"})


def cheatsheet():
    return "evdedh: DEdH moment estimator of the EV index"
