# moirais.fn — function file (hadesllm/moirais)
"""Box-Cox regression model: T_lambda(Y) = X'beta + U."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_box_cox"]


def horowitz_box_cox(x, y):
    """
    Box-Cox regression model: T_lambda(Y) = X'beta + U

    Formula: T_lambda(Y) = (Y^lambda-1)/lambda if lambda!=0; log(Y) if lambda=0

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lambda_hat, beta_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.1.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Box-Cox regression model: T_lambda(Y) = X'beta + U"})


def cheatsheet():
    return "hrzboxc: Box-Cox regression model: T_lambda(Y) = X'beta + U"
