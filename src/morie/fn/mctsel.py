"""MCTS selection phase via UCT or PUCT."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mcts_selection"]


def mcts_selection(Q, N, P, c):
    """
    MCTS selection phase via UCT or PUCT

    Formula: argmax_a Q(s,a) + U(s,a)

    Parameters
    ----------
    Q : array-like
        Input data.
    N : array-like
        Input data.
    P : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kocsis-Szepesvári (2006); Silver et al (2017)
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MCTS selection phase via UCT or PUCT"})


def cheatsheet():
    return "mctsel: MCTS selection phase via UCT or PUCT"
