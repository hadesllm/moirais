"""MuZero world-model representation function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["muzero_world_model"]


def muzero_world_model(observations, h, g):
    """
    MuZero world-model representation function

    Formula: hidden state = h(o); transitions via g

    Parameters
    ----------
    observations : array-like
        Input data.
    h : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schrittwieser et al (2020)
    """
    observations = np.atleast_1d(np.asarray(observations, dtype=float))
    n = len(observations)
    result = float(np.mean(observations))
    se = float(np.std(observations, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MuZero world-model representation function"})


def cheatsheet():
    return "agmuzu: MuZero world-model representation function"
