# moirais.fn — function file (hadesllm/moirais)
"""Mode collapse: GAN generator outputs limited variety."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mode_collapse"]


def geron_mode_collapse(samples):
    """
    Mode collapse: GAN generator outputs limited variety

    Formula: symptom: low sample diversity relative to data

    Parameters
    ----------
    samples : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: diversity_metric

    References
    ----------
    Géron Ch 18
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = len(samples)
    result = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mode collapse: GAN generator outputs limited variety"})


def cheatsheet():
    return "hmmdc: Mode collapse: GAN generator outputs limited variety"
