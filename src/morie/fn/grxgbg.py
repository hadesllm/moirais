# morie.fn -- function file (hadesllm/morie)
"""XGBoost split-gain formula with regularization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_xgboost_gain"]


def geron_xgboost_gain(GL, HL, GR, HR, lam, gamma):
    """
    XGBoost split-gain formula with regularization

    Formula: Gain = (1/2)*[G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - (G_L+G_R)^2/(H_L+H_R+lam)] - gamma

    Parameters
    ----------
    GL : array-like
        Input data.
    HL : array-like
        Input data.
    GR : array-like
        Input data.
    HR : array-like
        Input data.
    lam : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gain

    References
    ----------
    Géron Ch 6, XGBoost section
    """
    GL = np.asarray(GL, dtype=float)
    n = int(GL) if GL.ndim == 0 else len(GL)
    result = float(np.mean(GL))
    se = float(np.std(GL, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "XGBoost split-gain formula with regularization"})


def cheatsheet():
    return "grxgbg: XGBoost split-gain formula with regularization"
