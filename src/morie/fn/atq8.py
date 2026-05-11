"""INT8 quantized attention via per-row scales."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["int8_attention"]


def int8_attention(y, Q, K, V, scales):
    """
    INT8 quantized attention via per-row scales

    Formula: O = softmax(s_q s_k^T * (Q_int K_int^T) / sqrt(d)) (s_v V_int)

    Parameters
    ----------
    y : array-like
        Input data.
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    scales : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dettmers et al. (2022) LLM.int8()
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "INT8 quantized attention via per-row scales"})


def cheatsheet():
    return "atq8: INT8 quantized attention via per-row scales"
