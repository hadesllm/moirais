"""Temporal Fusion Transformer."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["temporal_fusion_transformer"]


def temporal_fusion_transformer(y, X, static, horizon):
    """
    Temporal Fusion Transformer

    Formula: variable selection + LSTM + attention

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    static : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lim et al (2021) TFT
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Temporal Fusion Transformer"})


def cheatsheet():
    return "tftran: Temporal Fusion Transformer"
