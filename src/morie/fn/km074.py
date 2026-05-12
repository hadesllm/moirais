r"""Dpo pref substituted.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_dpo_pref_substituted"]


def kamath_ch5_dpo_pref_substituted(pi_star, pi_ref, beta, Z):
    r"""
    Dpo pref substituted.

    Formula: p^*(y_w \succ y_l|x) = \sigma(\beta\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} + \beta\log Z(x) - \beta\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} - \beta\log Z(x))

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
    Kamath et al (2024), Ch 5, Eq 5.10, p. 210
    r"""
    pi_star = np.atleast_1d(np.asarray(pi_star, dtype=float))
    n = len(pi_star)
    result = float(np.mean(pi_star))
    se = float(np.std(pi_star, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dpo pref substituted."})


def cheatsheet():
    return "km074: Dpo pref substituted."
