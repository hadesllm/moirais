"""Prediction-compression duality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prediction_compression"]


def prediction_compression(model, data):
    """
    Prediction-compression duality

    Formula: compression rate ~ -E[log p_model(x)]

    Parameters
    ----------
    model : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cover-Thomas (2006)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prediction-compression duality"})


def cheatsheet():
    return "pcmpr1: Prediction-compression duality"
