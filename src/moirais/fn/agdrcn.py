"""AlphaZero Dirichlet concentration parameter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_dirichlet_concentration"]


def alphazero_dirichlet_concentration(avg_legal, scale):
    """
    AlphaZero Dirichlet concentration parameter

    Formula: alpha = 10/avg_legal_moves

    Parameters
    ----------
    avg_legal : array-like
        Input data.
    scale : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    avg_legal = np.atleast_1d(np.asarray(avg_legal, dtype=float))
    n = len(avg_legal)
    result = float(np.mean(avg_legal))
    se = float(np.std(avg_legal, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero Dirichlet concentration parameter"})


def cheatsheet():
    return "agdrcn: AlphaZero Dirichlet concentration parameter"
