"""Concept-drift detection (DDM)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["telemetry_drift"]


def telemetry_drift(error_stream):
    """
    Concept-drift detection (DDM)

    Formula: monitor error rate p_t and σ_t

    Parameters
    ----------
    error_stream : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gama et al (2004) DDM
    """
    error_stream = np.atleast_1d(np.asarray(error_stream, dtype=float))
    n = len(error_stream)
    result = float(np.mean(error_stream))
    se = float(np.std(error_stream, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Concept-drift detection (DDM)"})


def cheatsheet():
    return "telmt: Concept-drift detection (DDM)"
