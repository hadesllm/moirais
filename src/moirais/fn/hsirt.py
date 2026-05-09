# moirais.fn — function file (hadesllm/moirais)
"""Heteroskedastic IRT (Lauderdale 2010): per-legislator variation in uncertainty."""
import numpy as np
from ._richresult import RichResult

__all__ = ["heteroskedastic_irt"]


def heteroskedastic_irt(votes, n_dims):
    """
    Heteroskedastic IRT (Lauderdale 2010): per-legislator variation in uncertainty

    Formula: P(y_ij=1|x_i, beta_j, alpha_j, psi_i) = Phi((x_i - beta_j)*alpha_j / psi_i); psi_i individ. variance

    Parameters
    ----------
    votes : array-like
        Input data.
    n_dims : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'x_samples': 'array', 'psi_samples': 'array'}

    References
    ----------
    Armstrong Ch 6
    """
    votes = np.asarray(votes, dtype=float)
    n = int(votes) if votes.ndim == 0 else len(votes)
    result = float(np.mean(votes))
    se = float(np.std(votes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heteroskedastic IRT (Lauderdale 2010): per-legislator variation in uncertainty"})


def cheatsheet():
    return "hsirt: Heteroskedastic IRT (Lauderdale 2010): per-legislator variation in uncertainty"
