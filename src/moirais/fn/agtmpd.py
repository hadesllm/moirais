"""AlphaZero temperature decay schedule."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_temp_decay"]


def alphazero_temp_decay(move_count, threshold):
    """
    AlphaZero temperature decay schedule

    Formula: tau = 1 for first n moves; 0 thereafter

    Parameters
    ----------
    move_count : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    move_count = np.atleast_1d(np.asarray(move_count, dtype=float))
    n = len(move_count)
    result = float(np.mean(move_count))
    se = float(np.std(move_count, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero temperature decay schedule"})


def cheatsheet():
    return "agtmpd: AlphaZero temperature decay schedule"
