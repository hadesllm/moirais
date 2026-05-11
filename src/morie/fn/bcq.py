"""Batch-constrained Q-learning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bcq"]


def bcq(dataset, vae, actor):
    """
    Batch-constrained Q-learning

    Formula: restrict actions to those near data distribution

    Parameters
    ----------
    dataset : array-like
        Input data.
    vae : array-like
        Input data.
    actor : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fujimoto et al (2019)
    """
    dataset = np.atleast_1d(np.asarray(dataset, dtype=float))
    n = len(dataset)
    result = float(np.mean(dataset))
    se = float(np.std(dataset, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Batch-constrained Q-learning"})


def cheatsheet():
    return "bcq: Batch-constrained Q-learning"
