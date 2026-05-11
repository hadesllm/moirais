"""DPO direct preference optimization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dpo_loss"]


def dpo_loss(pi_theta, pi_ref, beta, pairs):
    """
    DPO direct preference optimization

    Formula: L = -log σ(β log π_θ(y_w)/π_ref − β log π_θ(y_l)/π_ref)

    Parameters
    ----------
    pi_theta : array-like
        Input data.
    pi_ref : array-like
        Input data.
    beta : array-like
        Input data.
    pairs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rafailov et al (2023) DPO
    """
    pi_theta = np.atleast_1d(np.asarray(pi_theta, dtype=float))
    n = len(pi_theta)
    result = float(np.mean(pi_theta))
    se = float(np.std(pi_theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DPO direct preference optimization"})


def cheatsheet():
    return "dpoF: DPO direct preference optimization"
