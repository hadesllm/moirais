"""AlphaZero vs Stockfish chess head-to-head."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_stockfish_baseline"]


def alphazero_stockfish_baseline(games, ladder):
    """
    AlphaZero vs Stockfish chess head-to-head

    Formula: win/draw/loss tally vs ELO ladder

    Parameters
    ----------
    games : array-like
        Input data.
    ladder : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2018)
    """
    games = np.atleast_1d(np.asarray(games, dtype=float))
    n = len(games)
    result = float(np.mean(games))
    se = float(np.std(games, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero vs Stockfish chess head-to-head"})


def cheatsheet():
    return "agstkb: AlphaZero vs Stockfish chess head-to-head"
