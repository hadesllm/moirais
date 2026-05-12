# morie.fn -- function file (hadesllm/morie)
"""O'Brien-Fleming interim analysis boundaries."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def interim_analysis(
    n_looks: int,
    *,
    alpha: float = 0.05,
    method: str = "obrien_fleming",
) -> DescriptiveResult:
    """
    Compute spending-function boundaries for interim analyses.

    Parameters
    ----------
    n_looks : int
        Number of planned interim looks (including final).
    alpha : float
        Overall type I error.
    method : str
        'obrien_fleming' or 'pocock'.

    Returns
    -------
    DescriptiveResult
        extra has 'boundaries', 'information_fractions'.

    References
    ----------
    O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing
    procedure for clinical trials. *Biometrics*, 35(3), 549-556.
    """
    if n_looks < 2:
        raise ValueError("n_looks must be >= 2.")

    fractions = np.arange(1, n_looks + 1) / n_looks
    boundaries = np.zeros(n_looks)

    if method == "obrien_fleming":
        for i, t in enumerate(fractions):
            boundaries[i] = stats.norm.ppf(1 - alpha / 2) / np.sqrt(t)
    elif method == "pocock":
        z = stats.norm.ppf(1 - alpha / (2 * n_looks))
        boundaries[:] = z
    else:
        raise ValueError(f"Unknown method: {method}")

    return DescriptiveResult(
        name="interim_analysis",
        value=float(boundaries[-1]),
        extra={
            "boundaries": boundaries.tolist(),
            "information_fractions": fractions.tolist(),
            "method": method,
            "alpha": alpha,
        },
    )


itrm = interim_analysis


def cheatsheet() -> str:
    return "interim_analysis({}) -> O'Brien-Fleming interim analysis boundaries."
