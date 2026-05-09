"""AlphaZero play log structured output."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_play_log"]


def alphazero_play_log(game, path):
    """
    AlphaZero play log structured output

    Formula: write game tree + outcomes

    Parameters
    ----------
    game : array-like
        Input data.
    path : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    game = np.atleast_1d(np.asarray(game, dtype=float))
    n = len(game)
    result = float(np.mean(game))
    se = float(np.std(game, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero play log structured output"})


def cheatsheet():
    return "agplog: AlphaZero play log structured output"
