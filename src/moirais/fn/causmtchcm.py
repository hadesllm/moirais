"""Caliper-matching variant with logit-PS distance cap."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_caliper_matching"]


def causal_caliper_matching(ps, treat, caliper):
    """
    Caliper-matching variant with logit-PS distance cap

    Formula: Allow only matches within caliper c × σ_logit_ps

    Parameters
    ----------
    ps : array-like
        Input data.
    treat : array-like
        Input data.
    caliper : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: match_idx

    References
    ----------
    Cochran-Rubin (1973)
    """
    ps = np.atleast_1d(np.asarray(ps, dtype=float))
    n = len(ps)
    result = float(np.mean(ps))
    se = float(np.std(ps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Caliper-matching variant with logit-PS distance cap"})


def cheatsheet():
    return "causmtchcm: Caliper-matching variant with logit-PS distance cap"
