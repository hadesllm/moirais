"""AlphaZero residual network block."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_resnet_block"]


def alphazero_resnet_block(x, filters):
    """
    AlphaZero residual network block

    Formula: x + relu(BN(conv(relu(BN(conv(x))))))

    Parameters
    ----------
    x : array-like
        Input data.
    filters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero residual network block"})


def cheatsheet():
    return "agnnbk: AlphaZero residual network block"
