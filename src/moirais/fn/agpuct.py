"""PUCT (Predictor Upper Confidence Tree) score for AlphaZero MCTS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_puct"]


def alphazero_puct(P, N, Q, c_puct):
    """
    PUCT (Predictor Upper Confidence Tree) score for AlphaZero MCTS

    Formula: U(s,a) = c_puct P(s,a) sqrt(N(s))/(1+N(s,a)) + Q(s,a)

    Parameters
    ----------
    P : array-like
        Input data.
    N : array-like
        Input data.
    Q : array-like
        Input data.
    c_puct : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017) Nature; Rosin (2011)
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PUCT (Predictor Upper Confidence Tree) score for AlphaZero MCTS"})


def cheatsheet():
    return "agpuct: PUCT (Predictor Upper Confidence Tree) score for AlphaZero MCTS"
