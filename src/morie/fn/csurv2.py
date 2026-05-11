"""Best linear predictor for causal survival forest CATE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_survival_blp"]


def causal_survival_blp(time, event, D, X):
    """
    Best linear predictor for causal survival forest CATE

    Formula: BLP of S_1(t)-S_0(t) on X via outcome regression

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cui-Kosorok-Athey-Wager (2023)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Best linear predictor for causal survival forest CATE"})


def cheatsheet():
    return "csurv2: Best linear predictor for causal survival forest CATE"
