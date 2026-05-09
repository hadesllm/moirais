"""Mean of a sum of two random processes equals sum of their means.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_mean_of_sum"]


def rangayyan_ch3_mean_of_sum(mu_x, mu_eta):
    """
    Mean of a sum of two random processes equals sum of their means.

    Formula: E[y] = mu_y = mu_x + mu_eta

    Parameters
    ----------
    mu_x : array-like
        Input data.
    mu_eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.13, p. 96
    """
    mu_x = np.atleast_1d(np.asarray(mu_x, dtype=float))
    n = len(mu_x)
    result = float(np.mean(mu_x))
    se = float(np.std(mu_x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean of a sum of two random processes equals sum of their means."})


def cheatsheet():
    return "rng013: Mean of a sum of two random processes equals sum of their means."
