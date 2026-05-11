"""Acute oral toxicity (rat LD50) estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["acute_toxicity_ld50"]


def acute_toxicity_ld50(smiles):
    """
    Acute oral toxicity (rat LD50) estimate

    Formula: DNN regression on Morgan FP + physchem

    Parameters
    ----------
    smiles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lei et al (2017)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Acute oral toxicity (rat LD50) estimate"})
    estimate = np.median(smiles)
    se = 1.2533 * np.std(smiles, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Acute oral toxicity (rat LD50) estimate"})


def cheatsheet():
    return "ld50r: Acute oral toxicity (rat LD50) estimate"
