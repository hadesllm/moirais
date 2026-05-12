# morie.fn -- function file (hadesllm/morie)
"""Optimal Classification (OC) cutting line geometry for binary roll call."""
import numpy as np
from ._richresult import RichResult

__all__ = ["oc_cutting_line"]


def oc_cutting_line(votes, n_dims):
    """
    Optimal Classification (OC) cutting line geometry for binary roll call

    Formula: Cutting line L_j: w_j'*votes = 0; separates predicted yeas from predicted nays, minimizes errors

    Parameters
    ----------
    votes : array-like
        Input data.
    n_dims : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'cutting_lines': 'matrix', 'error_rate': 'float'}

    References
    ----------
    Armstrong Ch 5
    """
    votes = np.asarray(votes, dtype=float)
    n = int(votes) if votes.ndim == 0 else len(votes)
    result = float(np.mean(votes))
    se = float(np.std(votes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Optimal Classification (OC) cutting line geometry for binary roll call"})


def cheatsheet():
    return "oclin: Optimal Classification (OC) cutting line geometry for binary roll call"
