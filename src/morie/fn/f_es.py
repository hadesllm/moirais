# morie.fn -- function file (hadesllm/morie)
"""Cohen's f effect size from eta-squared."""

import math

import numpy as np

from ._containers import ESRes


def cohens_f(eta2: float) -> ESRes:
    """Cohen's f from eta-squared.

    f = sqrt(eta2 / (1 - eta2))

    Parameters
    ----------
    eta2 : float
        Eta-squared value.

    Returns
    -------
    ESRes
    """
    f_val = math.sqrt(eta2 / (1 - eta2)) if eta2 < 1 else np.inf
    return ESRes(
        measure="Cohen's f",
        estimate=float(f_val),
    )


cf = cohens_f


def cheatsheet() -> str:
    return "cohens_f({}) -> Cohen's f effect size from eta-squared."
