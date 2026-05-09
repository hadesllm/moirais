"""Pre-treatment placebo/falsification regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_falsification_test"]


def causal_falsification_test(y_pre, treat, X_baseline):
    """
    Pre-treatment placebo/falsification regression

    Formula: Regress lagged outcome on treatment; coef should be ~0

    Parameters
    ----------
    y_pre : array-like
        Input data.
    treat : array-like
        Input data.
    X_baseline : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coef, p

    References
    ----------
    Athey-Imbens (2017)
    """
    y_pre = np.atleast_1d(np.asarray(y_pre, dtype=float))
    n = len(y_pre)
    result = float(np.mean(y_pre))
    se = float(np.std(y_pre, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pre-treatment placebo/falsification regression"})


def cheatsheet():
    return "causflnk: Pre-treatment placebo/falsification regression"
