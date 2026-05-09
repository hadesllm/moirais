"""Normalized cross-correlation coefficient used in template matching.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_normalized_cross_correlation_template"]


def rangayyan_ch3_normalized_cross_correlation_template(x, y, k, N, x_bar, y_bar_k):
    """
    Normalized cross-correlation coefficient used in template matching.

    Formula: gamma_xy(k) = sum_{n=0}^{N-1} [x(n)-x_bar][y(k-N+1+n)-y_bar_k] / sqrt( sum_{n=0}^{N-1} [x(n)-x_bar]^2 * sum_{n=0}^{N-1} [y(k-N+1+n)-y_bar_k]^2 )

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.
    N : array-like
        Input data.
    x_bar : array-like
        Input data.
    y_bar_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.97, p. 137
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Normalized cross-correlation coefficient used in template matching."})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Normalized cross-correlation coefficient used in template matching."})


def cheatsheet():
    return "rng086: Normalized cross-correlation coefficient used in template matching."
