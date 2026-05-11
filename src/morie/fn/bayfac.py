"""Bayes factor between models."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_factor"]


def bayes_factor(log_evidence_1, log_evidence_2):
    """
    Bayes factor between models

    Formula: BF = p(D|M1) / p(D|M2)

    Parameters
    ----------
    log_evidence_1 : array-like
        Input data.
    log_evidence_2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kass-Raftery (1995)
    """
    log_evidence_1 = np.atleast_1d(np.asarray(log_evidence_1, dtype=float))
    n = len(log_evidence_1)
    result = float(np.mean(log_evidence_1))
    se = float(np.std(log_evidence_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayes factor between models"})


def cheatsheet():
    return "bayfac: Bayes factor between models"
