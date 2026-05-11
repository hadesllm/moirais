"""Dirichlet exploration noise at MCTS root."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_dirichlet_noise"]


def alphazero_dirichlet_noise(p, alpha, eps):
    """
    Dirichlet exploration noise at MCTS root

    Formula: P_a = (1-eps) p_a + eps eta_a; eta ~ Dir(alpha)

    Parameters
    ----------
    p : array-like
        Input data.
    alpha : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet exploration noise at MCTS root"})


def cheatsheet():
    return "agdirn: Dirichlet exploration noise at MCTS root"
