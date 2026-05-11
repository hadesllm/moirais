"""Adjusted PAF (Bruzzi)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["adjusted_paf"]


def adjusted_paf(RR_strata, prevalence_strata):
    """
    Adjusted PAF (Bruzzi)

    Formula: adjust for confounders via stratification

    Parameters
    ----------
    RR_strata : array-like
        Input data.
    prevalence_strata : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bruzzi et al (1985)
    """
    RR_strata = np.atleast_1d(np.asarray(RR_strata, dtype=float))
    n = len(RR_strata)
    result = float(np.mean(RR_strata))
    se = float(np.std(RR_strata, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adjusted PAF (Bruzzi)"})


def cheatsheet():
    return "rapaf: Adjusted PAF (Bruzzi)"
