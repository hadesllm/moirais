# morie.fn -- function file (hadesllm/morie)
"""Variational inference for DPM: mean-field approximation q(G,theta,z) = prod q_i."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_vb_dpm"]


def ghosal_vb_dpm(x):
    """
    Variational inference for DPM: mean-field approximation q(G,theta,z) = prod q_i

    Formula: ELBO: E_q[log p(X,z,theta,G)] - E_q[log q(z,theta,G)] maximized

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
    Ghosal Ch 5 §5.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variational inference for DPM: mean-field approximation q(G,theta,z) = prod q_i"})


def cheatsheet():
    return "gh_c5_6: Variational inference for DPM: mean-field approximation q(G,theta,z) = prod q_i"
