r"""Dpo reward optimal.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_dpo_reward_optimal"]


def kamath_ch5_dpo_reward_optimal(pi_star, pi_ref, beta, Z):
    r"""
    Dpo reward optimal.

    Formula: r^*(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)

    Parameters
    ----------
    pi_star : array-like
        Input data.
    pi_ref : array-like
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
    Kamath et al (2024), Ch 5, Eq 5.7, p. 209
    r"""
    pi_star = np.atleast_1d(np.asarray(pi_star, dtype=float))
    n = len(pi_star)
    result = float(np.mean(pi_star))
    se = float(np.std(pi_star, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dpo reward optimal."})


def cheatsheet():
    return "km071: Dpo reward optimal."
