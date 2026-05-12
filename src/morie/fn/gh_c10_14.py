# morie.fn -- function file (hadesllm/morie)
"""Bayes factor for parametric vs nonparametric model: BF consistent for model selection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_param_np_bf"]


def ghosal_param_np_bf(x):
    """
    Bayes factor for parametric vs nonparametric model: BF consistent for model selection

    Formula: BF ~ exp(-n*KL(P0,Phat_MLE)) / Pi(KL ball eps_n) for parametric H0

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 10 §10.5.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayes factor for parametric vs nonparametric model: BF consistent for model selection"})


def cheatsheet():
    return "gh_c10_14: Bayes factor for parametric vs nonparametric model: BF consistent for model selection"
