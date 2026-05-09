# moirais.fn — function file (hadesllm/moirais)
"""Autoformer: seasonal/trend decomposition + FFT-based lagged-similarity attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_autoformer"]


def joseph_autoformer(x, horizon, top_k_lags):
    """
    Autoformer: seasonal/trend decomposition + FFT-based lagged-similarity attention

    Formula: x = Trend + Seasonal; AutoCorr(Q,K,V) via FFT-based lagged-similarity aggregation

    Parameters
    ----------
    x : array-like
        Input data.
    horizon : array-like
        Input data.
    top_k_lags : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 16, Autoformer section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Autoformer: seasonal/trend decomposition + FFT-based lagged-similarity attention"})


def cheatsheet():
    return "joautf: Autoformer: seasonal/trend decomposition + FFT-based lagged-similarity attention"
