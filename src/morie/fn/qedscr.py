"""Quantitative Estimate of Drug-likeness (QED)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["qed_drug_likeness"]


def qed_drug_likeness(smiles, weights):
    """
    Quantitative Estimate of Drug-likeness (QED)

    Formula: weighted geometric mean of 8 desirability functions

    Parameters
    ----------
    smiles : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bickerton et al (2012)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Quantitative Estimate of Drug-likeness (QED)"})
    estimate = np.median(smiles)
    se = 1.2533 * np.std(smiles, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Quantitative Estimate of Drug-likeness (QED)"})


def cheatsheet():
    return "qedscr: Quantitative Estimate of Drug-likeness (QED)"
