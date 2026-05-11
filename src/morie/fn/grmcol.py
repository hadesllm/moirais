# morie.fn — function file (hadesllm/morie)
"""Mode-collapse metric: fraction of real-distribution modes missed by generator samples."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gan_mode_collapse_metric"]


def geron_gan_mode_collapse_metric(samples, true_modes):
    """
    Mode-collapse metric: fraction of real-distribution modes missed by generator samples

    Formula: coverage = |modes_hit| / |modes_true|; 1 - coverage = mode collapse rate

    Parameters
    ----------
    samples : array-like
        Input data.
    true_modes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coverage

    References
    ----------
    Géron Ch 18, Training Difficulties / Mode Collapse section
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = len(samples)
    result = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mode-collapse metric: fraction of real-distribution modes missed by generator samples"})


def cheatsheet():
    return "grmcol: Mode-collapse metric: fraction of real-distribution modes missed by generator samples"
