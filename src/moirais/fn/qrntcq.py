"""Quarantine efficacy from delay distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["quarantine_efficacy"]


def quarantine_efficacy(incubation, quarantine_duration):
    """
    Quarantine efficacy from delay distribution

    Formula: effective infection prevented = integral over delay

    Parameters
    ----------
    incubation : array-like
        Input data.
    quarantine_duration : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ashcroft et al (2021)
    """
    incubation = np.atleast_1d(np.asarray(incubation, dtype=float))
    n = len(incubation)
    result = float(np.mean(incubation))
    se = float(np.std(incubation, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quarantine efficacy from delay distribution"})


def cheatsheet():
    return "qrntcq: Quarantine efficacy from delay distribution"
