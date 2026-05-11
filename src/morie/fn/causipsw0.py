"""ATO overlap weights for restricted target population."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_iptw_atoweights"]


def causal_iptw_atoweights(treat, ps):
    """
    ATO overlap weights for restricted target population

    Formula: w_i = treat*(1-e) + (1-treat)*e

    Parameters
    ----------
    treat : array-like
        Input data.
    ps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights, ess

    References
    ----------
    Li-Morgan-Zaslavsky (2018)
    """
    treat = np.atleast_1d(np.asarray(treat, dtype=float))
    n = len(treat)
    result = float(np.mean(treat))
    se = float(np.std(treat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ATO overlap weights for restricted target population"})


def cheatsheet():
    return "causipsw0: ATO overlap weights for restricted target population"
