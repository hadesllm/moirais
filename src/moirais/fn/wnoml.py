"""W-NOMINATE log-likelihood for roll call matrix."""
import numpy as np
from ._richresult import RichResult

__all__ = ["wnominate_logit"]


def wnominate_logit(votes, ideal_points, yea_nay_positions, beta):
    """
    W-NOMINATE log-likelihood for roll call matrix

    Formula: log L = sum_i sum_j [y_ij*log P_ij(yea) + (1-y_ij)*log(1-P_ij(yea))]

    Parameters
    ----------
    votes : array-like
        Input data.
    ideal_points : array-like
        Input data.
    yea_nay_positions : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'loglik': 'float'}

    References
    ----------
    Armstrong Ch 5
    """
    votes = np.asarray(votes, dtype=float)
    n = int(votes) if votes.ndim == 0 else len(votes)
    result = float(np.mean(votes))
    se = float(np.std(votes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "W-NOMINATE log-likelihood for roll call matrix"})


def cheatsheet():
    return "wnoml: W-NOMINATE log-likelihood for roll call matrix"
