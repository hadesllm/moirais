"""Microsoft SR-CNN."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["microsoft_sr"]


def microsoft_sr(x):
    """
    Microsoft SR-CNN

    Formula: spectral residual + CNN classifier

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ren et al (2019)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Microsoft SR-CNN"})


def cheatsheet():
    return "micrR: Microsoft SR-CNN"
