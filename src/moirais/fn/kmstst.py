# moirais.fn — function file (hadesllm/moirais)
"""StereoSet bias: fraction of stereotype-preferred vs anti-stereotype continuations."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_stereoset_bias"]


def kamath_stereoset_bias(stereo_probs, anti_probs):
    """
    StereoSet bias: fraction of stereotype-preferred vs anti-stereotype continuations

    Formula: SS_bias = |{i : p(s_stereo_i) > p(s_anti_i)}| / N

    Parameters
    ----------
    stereo_probs : array-like
        Input data.
    anti_probs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 6, StereoSet section
    """
    stereo_probs = np.atleast_1d(np.asarray(stereo_probs, dtype=float))
    n = len(stereo_probs)
    result = float(np.mean(stereo_probs))
    se = float(np.std(stereo_probs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "StereoSet bias: fraction of stereotype-preferred vs anti-stereotype continuations"})


def cheatsheet():
    return "kmstst: StereoSet bias: fraction of stereotype-preferred vs anti-stereotype continuations"
