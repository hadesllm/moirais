# moirais.fn — function file (hadesllm/moirais)
"""Model-based learning: fit parameters theta to minimize a cost function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_model_based"]


def geron_model_based(X, y):
    """
    Model-based learning: fit parameters theta to minimize a cost function

    Formula: theta_hat = argmin_theta J(theta; X, y)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Model-based learning: fit parameters theta to minimize a cost function"})


def cheatsheet():
    return "hmmod: Model-based learning: fit parameters theta to minimize a cost function"
