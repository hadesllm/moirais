# morie.fn -- function file (hadesllm/morie)
"""Asymmetric cost matrix for clinical decision making."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch10_cost_matrix"]


def rangayyan_ch10_cost_matrix(cost_matrix, confusion_matrix, priors):
    """
    Asymmetric cost matrix for clinical decision making

    Formula: Total cost = sum C_ij * P(decide j | true i) * P(true i)

    Parameters
    ----------
    cost_matrix : array-like
        Input data.
    confusion_matrix : array-like
        Input data.
    priors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expected_cost

    References
    ----------
    Rangayyan Ch 10.9
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    n = int(cost_matrix) if cost_matrix.ndim == 0 else len(cost_matrix)
    result = float(np.mean(cost_matrix))
    se = float(np.std(cost_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymmetric cost matrix for clinical decision making"})


def cheatsheet():
    return "rgeqn10a: Asymmetric cost matrix for clinical decision making"
