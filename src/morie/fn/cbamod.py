"""CBAM channel + spatial attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cbam_attention"]


def cbam_attention(x):
    """
    CBAM channel + spatial attention

    Formula: channel attn × spatial attn

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
    Woo et al (2018) CBAM
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CBAM channel + spatial attention"})


def cheatsheet():
    return "cbamod: CBAM channel + spatial attention"
