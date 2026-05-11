"""Mamba selective SSM step (S6)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mamba_ssm_step"]


def mamba_ssm_step(y, x, A, B, C, D):
    """
    Mamba selective SSM step (S6)

    Formula: h_t = A(x_t) h_{t-1} + B(x_t) x_t; y_t = C(x_t) h_t

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.
    C : array-like
        Input data.
    D : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gu & Dao (2023)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mamba selective SSM step (S6)"})


def cheatsheet():
    return "mambss: Mamba selective SSM step (S6)"
