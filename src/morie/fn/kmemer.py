# morie.fn -- function file (hadesllm/morie)
"""Emergent ability metric: step-function improvement past a scale threshold."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_emergent_abilities"]


def kamath_emergent_abilities(scales, scores, threshold):
    """
    Emergent ability metric: step-function improvement past a scale threshold

    Formula: metric(N) = H(N - N_threshold) * f(N)  (roughly piecewise step)

    Parameters
    ----------
    scales : array-like
        Input data.
    scores : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: emergent_score

    References
    ----------
    Kamath Ch 1, Emergent Abilities section
    """
    scales = np.atleast_1d(np.asarray(scales, dtype=float))
    n = len(scales)
    result = float(np.mean(scales))
    se = float(np.std(scales, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Emergent ability metric: step-function improvement past a scale threshold"})


def cheatsheet():
    return "kmemer: Emergent ability metric: step-function improvement past a scale threshold"
