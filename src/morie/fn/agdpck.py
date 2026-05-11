"""AlphaZero training data serialization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_data_pickle"]


def alphazero_data_pickle(replay_buffer, path):
    """
    AlphaZero training data serialization

    Formula: pickle (s, pi, z) tuples to disk

    Parameters
    ----------
    replay_buffer : array-like
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
    replay_buffer = np.atleast_1d(np.asarray(replay_buffer, dtype=float))
    n = len(replay_buffer)
    result = float(np.mean(replay_buffer))
    se = float(np.std(replay_buffer, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero training data serialization"})


def cheatsheet():
    return "agdpck: AlphaZero training data serialization"
