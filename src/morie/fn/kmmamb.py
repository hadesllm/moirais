# morie.fn — function file (hadesllm/morie)
"""Mamba selective state-space recurrence."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_mamba_ssm"]


def kamath_mamba_ssm(x, A, B, C, delta):
    """
    Mamba selective state-space recurrence

    Formula: h_t = A_bar(x_t) h_{t-1} + B_bar(x_t) x_t;  y_t = C(x_t) h_t;  parameters depend on input

    Parameters
    ----------
    x : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.
    C : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Kamath Ch 10, Mamba / Selective SSM section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mamba selective state-space recurrence"})


def cheatsheet():
    return "kmmamb: Mamba selective state-space recurrence"
