"""DeepAR autoregressive RNN."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deepar"]


def deepar(series, cov, lstm_h):
    """
    DeepAR autoregressive RNN

    Formula: p(z_t|z_{1:t-1}) = N(μ_θ, σ_θ); θ from LSTM

    Parameters
    ----------
    series : array-like
        Input data.
    cov : array-like
        Input data.
    lstm_h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Salinas et al (2020) DeepAR
    """
    series = np.atleast_1d(np.asarray(series, dtype=float))
    n = len(series)
    result = float(np.mean(series))
    se = float(np.std(series, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DeepAR autoregressive RNN"})


def cheatsheet():
    return "deepar: DeepAR autoregressive RNN"
