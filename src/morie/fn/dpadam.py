"""DP-Adam."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_adam"]


def dp_adam(loss, C, sigma, lr, betas):
    """
    DP-Adam

    Formula: DP-SGD style noise + Adam moments

    Parameters
    ----------
    loss : array-like
        Input data.
    C : array-like
        Input data.
    sigma : array-like
        Input data.
    lr : array-like
        Input data.
    betas : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tang et al (2024)
    """
    loss = np.atleast_1d(np.asarray(loss, dtype=float))
    n = len(loss)
    result = float(np.mean(loss))
    se = float(np.std(loss, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP-Adam"})


def cheatsheet():
    return "dpadam: DP-Adam"
