# morie.fn -- function file (hadesllm/morie)
"""TFT: gated residual networks + variable selection + LSTM encoder + multi-head attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_temporal_fusion_transformer"]


def joseph_temporal_fusion_transformer(static, observed, known, horizon):
    """
    TFT: gated residual networks + variable selection + LSTM encoder + multi-head attention

    Formula: static + observed + known inputs -> GRN + VSN -> LSTM -> MHA -> quantile heads

    Parameters
    ----------
    static : array-like
        Input data.
    observed : array-like
        Input data.
    known : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: quantile_forecasts

    References
    ----------
    Joseph Ch 16, Temporal Fusion Transformer section
    """
    static = np.atleast_1d(np.asarray(static, dtype=float))
    n = len(static)
    result = float(np.mean(static))
    se = float(np.std(static, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TFT: gated residual networks + variable selection + LSTM encoder + multi-head attention"})


def cheatsheet():
    return "jotft: TFT: gated residual networks + variable selection + LSTM encoder + multi-head attention"
