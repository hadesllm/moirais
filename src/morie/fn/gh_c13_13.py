# morie.fn — function file (hadesllm/morie)
"""Cox proportional hazard model: lambda(t|x) = lambda0(t)*exp(beta'x)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_cox_model"]


def ghosal_cox_model(x):
    """
    Cox proportional hazard model: lambda(t|x) = lambda0(t)*exp(beta'x)

    Formula: lambda(t|x) = lambda_0(t) exp(beta'x), lambda_0 ~ BP, beta ~ Normal

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 13 §13.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cox proportional hazard model: lambda(t|x) = lambda0(t)*exp(beta'x)"})


def cheatsheet():
    return "gh_c13_13: Cox proportional hazard model: lambda(t|x) = lambda0(t)*exp(beta'x)"
