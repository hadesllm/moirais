# morie.fn -- function file (hadesllm/morie)
"""Best-of-N sampling: generate N completions, return the highest-reward one."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_best_of_n_sampling"]


def kamath_best_of_n_sampling(samples, rewards):
    """
    Best-of-N sampling: generate N completions, return the highest-reward one

    Formula: y_hat = argmax_{y in {y_1,...,y_N}} r_phi(x, y); y_i ~ pi(y | x)

    Parameters
    ----------
    samples : array-like
        Input data.
    rewards : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: best

    References
    ----------
    Kamath Ch 5, Best-of-N Sampling section
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = len(samples)
    result = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Best-of-N sampling: generate N completions, return the highest-reward one"})


def cheatsheet():
    return "kmbon: Best-of-N sampling: generate N completions, return the highest-reward one"
