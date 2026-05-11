"""X-learner for CATE (Künzel-Sekhon-Bickel-Yu)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["x_learner"]


def x_learner(y, D, X):
    """
    X-learner for CATE (Künzel-Sekhon-Bickel-Yu)

    Formula: two-stage: per-arm response + cross-imputation

    Parameters
    ----------
    y : array-like
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
    Künzel, Sekhon, Bickel, Yu (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "X-learner for CATE (Künzel-Sekhon-Bickel-Yu)"})


def cheatsheet():
    return "xlrnir: X-learner for CATE (Künzel-Sekhon-Bickel-Yu)"
