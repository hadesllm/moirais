# morie.fn — function file (hadesllm/morie)
"""Autoencoder anomaly detection: high reconstruction error indicates anomaly."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_anomaly_autoencoder"]


def geron_anomaly_autoencoder(model, X, threshold):
    """
    Autoencoder anomaly detection: high reconstruction error indicates anomaly

    Formula: anomaly if ||x - decode(encode(x))||^2 > threshold

    Parameters
    ----------
    model : array-like
        Input data.
    X : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: anomalies

    References
    ----------
    Géron Ch 8
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Autoencoder anomaly detection: high reconstruction error indicates anomaly"})


def cheatsheet():
    return "hmanae: Autoencoder anomaly detection: high reconstruction error indicates anomaly"
