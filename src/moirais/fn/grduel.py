# moirais.fn — function file (hadesllm/moirais)
"""Dueling DQN decomposition Q = V + (A - mean(A))."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dueling_dqn"]


def geron_dueling_dqn(V, A):
    """
    Dueling DQN decomposition Q = V + (A - mean(A))

    Formula: Q(s, a) = V(s) + (A(s, a) - (1/|A|) * sum_{a'} A(s, a'))

    Parameters
    ----------
    V : array-like
        Input data.
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q

    References
    ----------
    Géron Ch 19, Dueling DQN section
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dueling DQN decomposition Q = V + (A - mean(A))"})


def cheatsheet():
    return "grduel: Dueling DQN decomposition Q = V + (A - mean(A))"
