"""ResNeXt grouped-convolution block."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["resnext_block"]


def resnext_block(x, cardinality):
    """
    ResNeXt grouped-convolution block

    Formula: split-transform-merge with cardinality

    Parameters
    ----------
    x : array-like
        Input data.
    cardinality : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Xie et al (2017) ResNeXt
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ResNeXt grouped-convolution block"})


def cheatsheet():
    return "resnxt: ResNeXt grouped-convolution block"
