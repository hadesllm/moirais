# morie.fn -- function file (hadesllm/morie)
"""RLHF reward shaping with KL penalty against a reference policy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_kl_reward_shaping"]


def kamath_kl_reward_shaping(r_phi, kl_divergence, beta):
    """
    RLHF reward shaping with KL penalty against a reference policy

    Formula: r_shaped(x, y) = r_phi(x, y) - beta * KL( pi_theta(.|x) || pi_ref(.|x) )

    Parameters
    ----------
    r_phi : array-like
        Input data.
    kl_divergence : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: r_shaped

    References
    ----------
    Kamath Ch 5, KL Divergence Penalty section
    """
    r_phi = np.atleast_1d(np.asarray(r_phi, dtype=float))
    n = len(r_phi)
    result = float(np.mean(r_phi))
    se = float(np.std(r_phi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RLHF reward shaping with KL penalty against a reference policy"})


def cheatsheet():
    return "kmklr: RLHF reward shaping with KL penalty against a reference policy"
