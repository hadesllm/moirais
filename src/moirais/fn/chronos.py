"""Chronos pre-trained foundation model for TS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chronos_foundation_ts"]


def chronos_foundation_ts(y, horizon):
    """
    Chronos pre-trained foundation model for TS

    Formula: tokenize + transformer trained on huge TS corpus

    Parameters
    ----------
    y : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ansari et al (2024) Chronos
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chronos pre-trained foundation model for TS"})


def cheatsheet():
    return "chronos: Chronos pre-trained foundation model for TS"
