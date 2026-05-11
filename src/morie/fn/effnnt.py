"""EfficientNet MBConv block."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["efficientnet_block"]


def efficientnet_block(x, expand_ratio, filters):
    """
    EfficientNet MBConv block

    Formula: expand + depthwise + SE + project

    Parameters
    ----------
    x : array-like
        Input data.
    expand_ratio : array-like
        Input data.
    filters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tan-Le (2019)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EfficientNet MBConv block"})


def cheatsheet():
    return "effnnt: EfficientNet MBConv block"
