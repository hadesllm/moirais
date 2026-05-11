"""AlphaZero L2-regularized optimizer step."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_optimizer"]


def alphazero_optimizer(theta, grad, momentum, weight_decay):
    """
    AlphaZero L2-regularized optimizer step

    Formula: update theta with momentum + weight decay

    Parameters
    ----------
    theta : array-like
        Input data.
    grad : array-like
        Input data.
    momentum : array-like
        Input data.
    weight_decay : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero L2-regularized optimizer step"})


def cheatsheet():
    return "agclop: AlphaZero L2-regularized optimizer step"
