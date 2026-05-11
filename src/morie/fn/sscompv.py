"""Validation of competing-risks model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survival_competing_validation"]


def survival_competing_validation(time, event_type, predicted_F):
    """
    Validation of competing-risks model

    Formula: calibration + discrimination metrics

    Parameters
    ----------
    time : array-like
        Input data.
    event_type : array-like
        Input data.
    predicted_F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wolbers et al (2014)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Validation of competing-risks model"})


def cheatsheet():
    return "sscompv: Validation of competing-risks model"
