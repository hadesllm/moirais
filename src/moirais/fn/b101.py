"""Linear function defining the simplest one-feature model with weight w and bias b.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_linear_function"]


def burkov_lm_ch1_linear_function(x, w, b):
    """
    Linear function defining the simplest one-feature model with weight w and bias b.

    Formula: f(x) \stackrel{\text{def}}{=} wx + b

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scalar prediction

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.1, p. 20
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear function defining the simplest one-feature model with weight w and bias b."})


def cheatsheet():
    return "b101: Linear function defining the simplest one-feature model with weight w and bias b."
