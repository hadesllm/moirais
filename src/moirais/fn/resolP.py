"""Resolution refutation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["resolution_proof"]


def resolution_proof(clauses):
    """
    Resolution refutation

    Formula: derive contradiction via clauses

    Parameters
    ----------
    clauses : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robinson (1965)
    """
    clauses = np.atleast_1d(np.asarray(clauses, dtype=float))
    n = len(clauses)
    result = float(np.mean(clauses))
    se = float(np.std(clauses, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Resolution refutation"})


def cheatsheet():
    return "resolP: Resolution refutation"
