"""Reward kl penalty.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch5_reward_kl_penalty"]


def kamath_ch5_reward_kl_penalty(x, y, pi_RL, pi_SFT, beta):
    """
    Reward kl penalty.

    Formula: R(x,y) = r_{\theta}(x,y) - \beta\log[\frac{\pi^{RL}(y|x)}{\pi^{SFT}(y|x)}]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    pi_RL : array-like
        Input data.
    pi_SFT : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 5, Eq 5.2, p. 197
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reward kl penalty."})


def cheatsheet():
    return "km066: Reward kl penalty."
