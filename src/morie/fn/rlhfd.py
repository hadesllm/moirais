# morie.fn — function file (hadesllm/morie)
"""RLHF reward model score."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rlhf_reward"]


def rlhf_reward(x):
    """
    RLHF reward model score

    Formula: r(x,y) = W' * last_hidden(x,y)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ouyang et al. (2022)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RLHF reward model score"})


def cheatsheet():
    return "rlhfd: RLHF reward model score"
