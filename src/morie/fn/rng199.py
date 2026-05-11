"""Correlation coefficient as normalized dot product of two signals.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_correlation_coefficient_normalized_dot"]


def rangayyan_ch4_correlation_coefficient_normalized_dot(x, y, N):
    """
    Correlation coefficient as normalized dot product of two signals.

    Formula: gamma_xy = sum_{n=0}^{N-1} x(n)*y(n) / sqrt( sum_{n=0}^{N-1} x^2(n) * sum_{n=0}^{N-1} y^2(n) )

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.25, p. 229
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Correlation coefficient as normalized dot product of two signals."})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Correlation coefficient as normalized dot product of two signals."})


def cheatsheet():
    return "rng199: Correlation coefficient as normalized dot product of two signals."
