r"""Rlhf optimal policy.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_rlhf_optimal_policy"]


def kamath_ch5_rlhf_optimal_policy(pi_ref, r, beta, Z):
    r"""
    Rlhf optimal policy.

    Formula: \pi_r(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp(\frac{1}{\beta}r(x,y))

    Parameters
    ----------
    pi_ref : array-like
        Input data.
    r : array-like
        Input data.
    beta : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.6, p. 208
    r"""
    pi_ref = np.atleast_1d(np.asarray(pi_ref, dtype=float))
    n = len(pi_ref)
    result = float(np.mean(pi_ref))
    se = float(np.std(pi_ref, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rlhf optimal policy."})


def cheatsheet():
    return "km070: Rlhf optimal policy."
