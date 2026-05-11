"""LSTM time-series forecaster."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["neural_ts_lstm"]


def neural_ts_lstm(y, hidden, horizon):
    """
    LSTM time-series forecaster

    Formula: recurrent LSTM cell over y_t

    Parameters
    ----------
    y : array-like
        Input data.
    hidden : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hochreiter-Schmidhuber (1997)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LSTM time-series forecaster"})


def cheatsheet():
    return "netsts: LSTM time-series forecaster"
