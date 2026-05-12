# morie.fn -- function file (hadesllm/morie)
"""Convert Cohen's d to NNT (Kraemer and Kupfer, 2006)."""

import numpy as np
import scipy.stats as stats


def d_to_nnt(d: float, base_rate: float = 0.5, cdf=None) -> float:
    """Convert Cohen's d to NNT given a base rate.

    Uses the Kraemer and Kupfer (2006) formula:
    NNT = 1 / (Phi(d/2 + Phi^-1(CER)) - CER)

    Parameters
    ----------
    d : float
    base_rate : float, default 0.5
        Control event rate.

    Returns
    -------
    float
    """
    z_cer = stats.norm.ppf(base_rate)
    p_treat = stats.norm.cdf(d + z_cer)
    rd = p_treat - base_rate
    return 1 / abs(rd) if abs(rd) > 0 else np.inf


d2nnt = d_to_nnt


def cheatsheet() -> str:
    return "d_to_nnt({}) -> Convert Cohen's d to NNT (Kraemer and Kupfer, 2006)."
