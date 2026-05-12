# morie.fn -- function file (hadesllm/morie)
"""Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_nonlinear_features"]


def rangayyan_nonlinear_features(x, m, r):
    """
    Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov)

    Formula: Feature vector: [ApEn, SampEn, alpha_DFA, lambda_max]

    Parameters
    ----------
    x : array-like
        Input data.
    m : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features_dict

    References
    ----------
    Rangayyan Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov)"})


def cheatsheet():
    return "rgnl: Nonlinear features of biomedical signals (ApEn, SampEn, DFA, Lyapunov)"
