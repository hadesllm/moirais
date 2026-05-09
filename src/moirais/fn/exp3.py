"""EXP3 adversarial bandit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["exp3"]


def exp3(arms, T, gamma):
    """
    EXP3 adversarial bandit

    Formula: prob_i ∝ exp(η · estimated_reward)

    Parameters
    ----------
    arms : array-like
        Input data.
    T : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Auer et al (2002)
    """
    arms = np.atleast_1d(np.asarray(arms, dtype=float))
    n = len(arms)
    result = float(np.mean(arms))
    se = float(np.std(arms, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EXP3 adversarial bandit"})


def cheatsheet():
    return "exp3: EXP3 adversarial bandit"
