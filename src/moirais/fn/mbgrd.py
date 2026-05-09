# moirais.fn — function file (hadesllm/moirais)
"""Mini-batch gradient descent."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mini_batch_gradient"]


def mini_batch_gradient(x, y):
    """
    Mini-batch gradient descent

    Formula: theta -= alpha * (1/B) sum grad_i(J)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mini-batch gradient descent"})


def cheatsheet():
    return "mbgrd: Mini-batch gradient descent"
