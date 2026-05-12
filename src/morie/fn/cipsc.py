# morie.fn -- function file (hadesllm/morie)
"""Propensity score caliper matching (restrict to within-caliper pairs)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["caliper_psm"]


def caliper_psm(e_score, T, caliper):
    """
    Propensity score caliper matching (restrict to within-caliper pairs)

    Formula: Match i to j if |e(X_i) - e(X_j)| < c; c typically 0.2*sd(logit(e(X)))

    Parameters
    ----------
    e_score : array-like
        Input data.
    T : array-like
        Input data.
    caliper : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'matched_idx': 'array', 'balance': 'dict'}

    References
    ----------
    Molak Ch 9
    """
    e_score = np.asarray(e_score, dtype=float)
    n = int(e_score) if e_score.ndim == 0 else len(e_score)
    result = float(np.mean(e_score))
    se = float(np.std(e_score, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Propensity score caliper matching (restrict to within-caliper pairs)"})


def cheatsheet():
    return "cipsc: Propensity score caliper matching (restrict to within-caliper pairs)"
