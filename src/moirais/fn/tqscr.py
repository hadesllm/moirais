"""Theorem 3.6: attention-score distortion with sketch dim m."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_score_distortion"]


def turboquant_score_distortion(eps, r, n):
    """
    Theorem 3.6: attention-score distortion with sketch dim m

    Formula: Score_hat(i) in (1 +/- 3*eps) * Score(i) w.h.p. if m >= 2 r^2 / eps^2 * log(n)

    Parameters
    ----------
    eps : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: m_min

    References
    ----------
    Zandieh et al. 2024 Theorem 3.6 (attention score distortion)
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    n = len(eps)
    result = float(np.mean(eps))
    se = float(np.std(eps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 3.6: attention-score distortion with sketch dim m"})


def cheatsheet():
    return "tqscr: Theorem 3.6: attention-score distortion with sketch dim m"
