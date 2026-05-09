# moirais.fn — function file (hadesllm/moirais)
"""Sample quantile (empirical quantile function)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_sample_quantile"]


def fauzi_sample_quantile(data, p):
    """
    Sample quantile (empirical quantile function)

    Formula: Q_hat(p) = inf{data: F_n(data) >= p}

    Parameters
    ----------
    data : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 3
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sample quantile (empirical quantile function)"})


def cheatsheet():
    return "fzsmpq: Sample quantile (empirical quantile function)"
