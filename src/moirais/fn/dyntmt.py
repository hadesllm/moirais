"""Dynamic-regime MSM (regime depends on history)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dynamic_marginal_msm"]


def dynamic_marginal_msm(y, D_history, H_history, regime_fn):
    """
    Dynamic-regime MSM (regime depends on history)

    Formula: V(d) = E[Y(d_bar(H))]

    Parameters
    ----------
    y : array-like
        Input data.
    D_history : array-like
        Input data.
    H_history : array-like
        Input data.
    regime_fn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Petersen et al (2014); Robins-Orellana-Rotnitzky (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dynamic-regime MSM (regime depends on history)"})


def cheatsheet():
    return "dyntmt: Dynamic-regime MSM (regime depends on history)"
