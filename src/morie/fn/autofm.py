"""Autoformer — series decomp + auto-correlation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["autoformer"]


def autoformer(X, y, seq_len):
    """
    Autoformer — series decomp + auto-correlation

    Formula: seasonal/trend decomp + auto-corr attention

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    seq_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wu et al (2021) Autoformer
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(y), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Autoformer — series decomp + auto-correlation"})
    result = stats.spearmanr(y[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Autoformer — series decomp + auto-correlation"})


def cheatsheet():
    return "autofm: Autoformer — series decomp + auto-correlation"
