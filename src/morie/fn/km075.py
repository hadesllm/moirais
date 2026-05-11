"""Dpo pref simplified.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_dpo_pref_simplified"]


def kamath_ch5_dpo_pref_simplified(pi_star, pi_ref, beta):
    """
    Dpo pref simplified.

    Formula: p^*(y_w \succ y_l|x) = \sigma(\beta\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)})

    Parameters
    ----------
    pi_star : array-like
        Input data.
    pi_ref : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.11, p. 210
    """
    pi_star = np.atleast_1d(np.asarray(pi_star, dtype=float))
    n = len(pi_star)
    result = float(np.mean(pi_star))
    se = float(np.std(pi_star, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dpo pref simplified."})


def cheatsheet():
    return "km075: Dpo pref simplified."
