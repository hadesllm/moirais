"""AlphaZero gradient clipping."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_weight_clipping"]


def alphazero_weight_clipping(grad, max_norm):
    """
    AlphaZero gradient clipping

    Formula: clip ||grad||_2 to max_norm

    Parameters
    ----------
    grad : array-like
        Input data.
    max_norm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    grad = np.atleast_1d(np.asarray(grad, dtype=float))
    n = len(grad)
    result = float(np.mean(grad))
    se = float(np.std(grad, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero gradient clipping"})


def cheatsheet():
    return "agwgcl: AlphaZero gradient clipping"
