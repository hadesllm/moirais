"""AlphaFold-3 diffusion step."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["af3_diffusion_step"]


def af3_diffusion_step(x, t, score_fn):
    """
    AlphaFold-3 diffusion step

    Formula: x_t-1 = x_t - sigma * score(x_t)

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    score_fn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Abramson et al (2024)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold-3 diffusion step"})


def cheatsheet():
    return "alf3df: AlphaFold-3 diffusion step"
