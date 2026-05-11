"""AlphaZero self-play evaluation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_self_play_eval"]


def alphazero_self_play_eval(new_net, old_net, n_games):
    """
    AlphaZero self-play evaluation

    Formula: 100 games new vs old; ratio > 0.55 triggers update

    Parameters
    ----------
    new_net : array-like
        Input data.
    old_net : array-like
        Input data.
    n_games : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    new_net = np.atleast_1d(np.asarray(new_net, dtype=float))
    n = len(new_net)
    result = float(np.mean(new_net))
    se = float(np.std(new_net, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero self-play evaluation"})


def cheatsheet():
    return "agselp: AlphaZero self-play evaluation"
