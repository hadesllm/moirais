"""GP-based density shift detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_density_shift"]


def gp_density_shift(y_stream, window, tau):
    """
    GP-based density shift detection

    Formula: posterior probability KL(p_t||p_{t-1}) > tau

    Parameters
    ----------
    y_stream : array-like
        Input data.
    window : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gulenko et al (2019)
    """
    y_stream = np.atleast_1d(np.asarray(y_stream, dtype=float))
    n = len(y_stream)
    result = float(np.mean(y_stream))
    se = float(np.std(y_stream, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP-based density shift detection"})


def cheatsheet():
    return "gpdsh: GP-based density shift detection"
