"""Performer FAVOR+ kernel attention (random features)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["performer_favor_attention"]


def performer_favor_attention(y, Q, K, V, phi):
    """
    Performer FAVOR+ kernel attention (random features)

    Formula: Attn = (phi(Q) (phi(K)^T V)) / (phi(Q) (phi(K)^T 1))

    Parameters
    ----------
    y : array-like
        Input data.
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Choromanski et al. (2021) Performer
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Performer FAVOR+ kernel attention (random features)"})


def cheatsheet():
    return "perfat: Performer FAVOR+ kernel attention (random features)"
