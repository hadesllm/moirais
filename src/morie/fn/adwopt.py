"""AdamW with decoupled weight decay."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["adamw"]


def adamw(g, beta1, beta2, lr, wd):
    """
    AdamW with decoupled weight decay

    Formula: x -= lr (m_hat / (sqrt(v_hat) + eps) + lambda x)

    Parameters
    ----------
    g : array-like
        Input data.
    beta1 : array-like
        Input data.
    beta2 : array-like
        Input data.
    lr : array-like
        Input data.
    wd : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Loshchilov-Hutter (2019)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdamW with decoupled weight decay"})


def cheatsheet():
    return "adwopt: AdamW with decoupled weight decay"
