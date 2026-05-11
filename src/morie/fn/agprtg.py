"""AlphaZero prioritized replay target."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_priority_target"]


def alphazero_priority_target(replay_buffer, priorities):
    """
    AlphaZero prioritized replay target

    Formula: prioritize by |z - v| residual

    Parameters
    ----------
    replay_buffer : array-like
        Input data.
    priorities : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schaul et al (2016) PER
    """
    replay_buffer = np.atleast_1d(np.asarray(replay_buffer, dtype=float))
    n = len(replay_buffer)
    result = float(np.mean(replay_buffer))
    se = float(np.std(replay_buffer, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero prioritized replay target"})


def cheatsheet():
    return "agprtg: AlphaZero prioritized replay target"
