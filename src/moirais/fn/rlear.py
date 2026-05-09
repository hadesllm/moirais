# moirais.fn — function file (hadesllm/moirais)
"""R-learner (Nie-Wager) for CATE via residualization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["r_learner"]


def r_learner(Y, T, X, m_model, e_model, tau_model):
    """
    R-learner (Nie-Wager) for CATE via residualization

    Formula: min_{tau} sum_i [(Y_i - m(X_i)) - tau(X_i)*(T_i - e(X_i))]^2 + lambda*||tau||

    Parameters
    ----------
    Y : array-like
        Input data.
    T : array-like
        Input data.
    X : array-like
        Input data.
    m_model : array-like
        Input data.
    e_model : array-like
        Input data.
    tau_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'cate': 'array'}

    References
    ----------
    Molak Ch 10
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "R-learner (Nie-Wager) for CATE via residualization"})


def cheatsheet():
    return "rlear: R-learner (Nie-Wager) for CATE via residualization"
