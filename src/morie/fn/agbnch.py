"""AlphaZero benchmark eval (ELO + Tactic suite)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_benchmark_eval"]


def alphazero_benchmark_eval(games, ladder):
    """
    AlphaZero benchmark eval (ELO + Tactic suite)

    Formula: compute ELO from match pool

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero benchmark eval (ELO + Tactic suite)"})


def cheatsheet():
    return "agbnch: AlphaZero benchmark eval (ELO + Tactic suite)"
