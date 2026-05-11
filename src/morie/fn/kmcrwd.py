# morie.fn — function file (hadesllm/morie)
"""CrowS-Pairs: pseudo-log-likelihood preference for stereotyping sentence in minimal pairs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_crowspairs_bias"]


def kamath_crowspairs_bias(stereo_pll, anti_pll):
    """
    CrowS-Pairs: pseudo-log-likelihood preference for stereotyping sentence in minimal pairs

    Formula: bias = |{i : PLL(s_stereo_i) > PLL(s_antistereo_i)}| / N

    Parameters
    ----------
    stereo_pll : array-like
        Input data.
    anti_pll : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 6, CrowS-Pairs section
    """
    stereo_pll = np.atleast_1d(np.asarray(stereo_pll, dtype=float))
    n = len(stereo_pll)
    result = float(np.mean(stereo_pll))
    se = float(np.std(stereo_pll, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CrowS-Pairs: pseudo-log-likelihood preference for stereotyping sentence in minimal pairs"})


def cheatsheet():
    return "kmcrwd: CrowS-Pairs: pseudo-log-likelihood preference for stereotyping sentence in minimal pairs"
