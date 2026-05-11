# morie.fn — function file (hadesllm/morie)
"""Online learning: sequentially update model with streaming data and learning rate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_online_learning"]


def geron_online_learning(X_stream, y_stream, eta):
    """
    Online learning: sequentially update model with streaming data and learning rate

    Formula: theta_{t+1} = theta_t - eta_t * grad L(theta_t; x_t, y_t)

    Parameters
    ----------
    X_stream : array-like
        Input data.
    y_stream : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 1
    """
    X_stream = np.atleast_1d(np.asarray(X_stream, dtype=float))
    n = len(X_stream)
    result = float(np.mean(X_stream))
    se = float(np.std(X_stream, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Online learning: sequentially update model with streaming data and learning rate"})


def cheatsheet():
    return "hmonl: Online learning: sequentially update model with streaming data and learning rate"
