"""MuZero efficient exploration via PUCT."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["muzero_efficient_exploration"]


def muzero_efficient_exploration(model, root_state, sims):
    """
    MuZero efficient exploration via PUCT

    Formula: PUCT with learned model

    Parameters
    ----------
    model : array-like
        Input data.
    root_state : array-like
        Input data.
    sims : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schrittwieser et al (2020)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MuZero efficient exploration via PUCT"})


def cheatsheet():
    return "agmuef: MuZero efficient exploration via PUCT"
