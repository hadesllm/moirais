"""AlphaZero loss variance estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_loss_var"]


def alphazero_loss_var(losses):
    """
    AlphaZero loss variance estimate

    Formula: running variance of (z-v)^2 + KL terms

    Parameters
    ----------
    losses : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2018)
    """
    losses = np.atleast_1d(np.asarray(losses, dtype=float))
    n = len(losses)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "AlphaZero loss variance estimate"})
    estimate = np.median(losses)
    se = 1.2533 * np.std(losses, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "AlphaZero loss variance estimate"})


def cheatsheet():
    return "aglnvr: AlphaZero loss variance estimate"
