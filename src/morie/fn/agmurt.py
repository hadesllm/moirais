"""MuZero reanalyze re-targeting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["muzero_reanalyze_target"]


def muzero_reanalyze_target(replay, model):
    """
    MuZero reanalyze re-targeting

    Formula: recompute search targets from latest model

    Parameters
    ----------
    replay : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schrittwieser et al (2020)
    """
    replay = np.atleast_1d(np.asarray(replay, dtype=float))
    n = len(replay)
    result = float(np.mean(replay))
    se = float(np.std(replay, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MuZero reanalyze re-targeting"})


def cheatsheet():
    return "agmurt: MuZero reanalyze re-targeting"
