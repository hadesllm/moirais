# moirais.fn — function file (hadesllm/moirais)
"""EWMA volatility (RiskMetrics)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ewma_volatility"]


def ewma_volatility(x):
    """
    EWMA volatility (RiskMetrics)

    Formula: sigma_t^2 = lambda*sigma_{t-1}^2 + (1-lambda)*r_{t-1}^2

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
    JP Morgan (1996)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EWMA volatility (RiskMetrics)"})


def cheatsheet():
    return "ewtma: EWMA volatility (RiskMetrics)"
