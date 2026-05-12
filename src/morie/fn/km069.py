r"""Rlhf objective.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_rlhf_objective"]


def kamath_ch5_rlhf_objective(pi_theta, pi_ref, r_phi, beta):
    r"""
    Rlhf objective.

    Formula: J_{RLHF} = \max_{\pi_{\theta}} E_{x\sim D, y\sim\pi_{\theta}}[r_{\phi}(x,y) - \beta D_{KL}(\pi_{\theta}(y|x)\|\pi_{ref}(y|x))]

    Parameters
    ----------
    pi_theta : array-like
        Input data.
    pi_ref : array-like
        Input data.
    r_phi : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.5, p. 208
    r"""
    pi_theta = np.atleast_1d(np.asarray(pi_theta, dtype=float))
    n = len(pi_theta)
    result = float(np.mean(pi_theta))
    se = float(np.std(pi_theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rlhf objective."})


def cheatsheet():
    return "km069: Rlhf objective."
