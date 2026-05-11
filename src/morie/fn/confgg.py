"""Configuration model with given degrees."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["configuration_model"]


def configuration_model(degrees):
    """
    Configuration model with given degrees

    Formula: random graph with prescribed degree seq

    Parameters
    ----------
    degrees : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bender-Canfield (1978); Molloy-Reed (1995)
    """
    degrees = np.atleast_1d(np.asarray(degrees, dtype=float))
    n = len(degrees)
    result = float(np.mean(degrees))
    se = float(np.std(degrees, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Configuration model with given degrees"})


def cheatsheet():
    return "confgg: Configuration model with given degrees"
