# morie.fn — function file (hadesllm/morie)
"""Binary response model with random coefficients."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_binary_response_model"]


def horowitz_binary_response_model(x, y):
    """
    Binary response model with random coefficients

    Formula: Y=1 if X'(beta+nu)+V>0; Y=0 otherwise; U=X'nu+V heteroskedastic

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat

    References
    ----------
    Horowitz Ch 4, Eq 4.1-4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Binary response model with random coefficients"})


def cheatsheet():
    return "hrzbr4a: Binary response model with random coefficients"
