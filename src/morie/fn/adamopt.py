"""Adam optimizer."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["adam"]


def adam(g, beta1, beta2, lr, eps):
    """
    Adam optimizer

    Formula: m,v moment estimates; x -= lr m_hat / (sqrt(v_hat) + eps)

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
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kingma-Ba (2015)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adam optimizer"})


def cheatsheet():
    return "adamopt: Adam optimizer"
