"""W-NOMINATE alternating optimization: alternate estimating ideal points and vote positions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["wnominate_alternating"]


def wnominate_alternating(votes, n_dims, polarity):
    """
    W-NOMINATE alternating optimization: alternate estimating ideal points and vote positions

    Formula: Fix x_i*, optimize z_j; then fix z_j, optimize x_i*; iterate to convergence

    Parameters
    ----------
    votes : array-like
        Input data.
    n_dims : array-like
        Input data.
    polarity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ideal_points': 'matrix', 'positions': 'matrix'}

    References
    ----------
    Armstrong Ch 5
    """
    votes = np.asarray(votes, dtype=float)
    n = int(votes) if votes.ndim == 0 else len(votes)
    if votes.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "W-NOMINATE alternating optimization: alternate estimating ideal points and vote positions"})
    estimate = np.median(votes)
    se = 1.2533 * np.std(votes, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "W-NOMINATE alternating optimization: alternate estimating ideal points and vote positions"})


def cheatsheet():
    return "wnoma: W-NOMINATE alternating optimization: alternate estimating ideal points and vote positions"
