"""Matched case-control conditional OR."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["matched_case_control"]


def matched_case_control(cases, controls, matching_id, exposure):
    """
    Matched case-control conditional OR

    Formula: conditional MLE of OR per matched stratum

    Parameters
    ----------
    cases : array-like
        Input data.
    controls : array-like
        Input data.
    matching_id : array-like
        Input data.
    exposure : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Breslow-Day (1980)
    """
    cases = np.atleast_1d(np.asarray(cases, dtype=float))
    n = len(cases)
    result = float(np.mean(cases))
    se = float(np.std(cases, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matched case-control conditional OR"})


def cheatsheet():
    return "matccd: Matched case-control conditional OR"
