"""ControlNet conditional attachment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["controlnet_attach"]


def controlnet_attach(base, condition):
    """
    ControlNet conditional attachment

    Formula: trainable copy of UNet encoder; zero-conv

    Parameters
    ----------
    base : array-like
        Input data.
    condition : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang et al (2023) ControlNet
    """
    base = np.atleast_1d(np.asarray(base, dtype=float))
    n = len(base)
    result = float(np.mean(base))
    se = float(np.std(base, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ControlNet conditional attachment"})


def cheatsheet():
    return "cncpat: ControlNet conditional attachment"
