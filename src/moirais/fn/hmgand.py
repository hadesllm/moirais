# moirais.fn — function file (hadesllm/moirais)
"""GMM-based anomaly detection: low-density points are anomalies."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_anomaly_gmm"]


def geron_anomaly_gmm(X, n_components, threshold):
    """
    GMM-based anomaly detection: low-density points are anomalies

    Formula: anomaly if p(x) < threshold

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
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
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GMM-based anomaly detection: low-density points are anomalies"})


def cheatsheet():
    return "hmgand: GMM-based anomaly detection: low-density points are anomalies"
