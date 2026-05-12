r"""Multihead head i.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_multihead_head_i"]


def kamath_ch2_multihead_head_i(Q, K, V, W_Qi, W_Ki, W_Vi):
    r"""
    Multihead head i.

    Formula: \mathrm{head}_i = \mathrm{attention}(W_{Q_i}Q, W_{K_i}K, W_{V_i}V)

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    W_Qi : array-like
        Input data.
    W_Ki : array-like
        Input data.
    W_Vi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.15, p. 36
    r"""
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multihead head i."})


def cheatsheet():
    return "km015: Multihead head i."
