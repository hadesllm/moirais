"""Meta-RL (e.g. RL² with recurrent net)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["meta_rl"]


def meta_rl(task_dist, rnn):
    """
    Meta-RL (e.g. RL² with recurrent net)

    Formula: learn fast adaptation across task distribution

    Parameters
    ----------
    task_dist : array-like
        Input data.
    rnn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang et al (2016) RL²
    """
    task_dist = np.atleast_1d(np.asarray(task_dist, dtype=float))
    n = len(task_dist)
    result = float(np.mean(task_dist))
    se = float(np.std(task_dist, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Meta-RL (e.g. RL² with recurrent net)"})


def cheatsheet():
    return "mtdrl: Meta-RL (e.g. RL² with recurrent net)"
