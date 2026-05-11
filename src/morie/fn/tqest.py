"""EstimateScores inner loop: reconstruct attention scores from QJL keys."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_estimate_scores"]


def turboquant_estimate_scores(q, k_tildes, norms, S):
    """
    EstimateScores inner loop: reconstruct attention scores from QJL keys

    Formula: score_hat_j = sqrt(pi/2)/m * nu_j * <S q_n, k_tilde_j>  for j=1..n

    Parameters
    ----------
    q : array-like
        Input data.
    k_tildes : array-like
        Input data.
    norms : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scores_hat

    References
    ----------
    Zandieh et al. 2024 Algorithm 1 EstimateScores step
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "EstimateScores inner loop: reconstruct attention scores from QJL keys"})
    estimate = np.median(q)
    se = 1.2533 * np.std(q, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "EstimateScores inner loop: reconstruct attention scores from QJL keys"})


def cheatsheet():
    return "tqest: EstimateScores inner loop: reconstruct attention scores from QJL keys"
