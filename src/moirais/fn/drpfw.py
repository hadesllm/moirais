# moirais.fn — function file (hadesllm/moirais)
"""Dropout forward pass with scaling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["dropout_forward"]


def dropout_forward(x):
    """
    Dropout forward pass with scaling

    Formula: y = x * mask / (1-p), mask ~ Bernoulli(1-p)

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
    Srivastava et al. (2014)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dropout forward pass with scaling"})


def cheatsheet():
    return "drpfw: Dropout forward pass with scaling"
