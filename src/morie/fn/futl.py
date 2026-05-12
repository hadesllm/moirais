# morie.fn -- function file (hadesllm/morie)
"""Futility stopping boundary."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def futility_boundary(
    n_looks: int,
    *,
    power: float = 0.80,
    alpha: float = 0.05,
) -> DescriptiveResult:
    """
    Compute futility (beta-spending) stopping boundaries.

    At each interim look, if the test statistic falls below the
    futility boundary, the trial may stop for futility.

    Parameters
    ----------
    n_looks : int
        Number of planned looks.
    power : float
        Target power.
    alpha : float
        Overall significance level.

    Returns
    -------
    DescriptiveResult
        extra has 'boundaries', 'information_fractions'.

    References
    ----------
    Pampallona, S., & Tsiatis, A. A. (1994). Group sequential designs
    for one-sided and two-sided hypothesis testing. *J Stat Plan
    Inference*, 42(1-2), 19-35.
    """
    if n_looks < 2:
        raise ValueError("n_looks must be >= 2.")

    fractions = np.arange(1, n_looks + 1) / n_looks
    z_final = stats.norm.ppf(1 - alpha / 2)
    boundaries = np.zeros(n_looks)

    for i, t in enumerate(fractions):
        boundaries[i] = z_final * np.sqrt(t) - stats.norm.ppf(power) * (1 - t) / np.sqrt(t)

    return DescriptiveResult(
        name="futility_boundary",
        value=float(boundaries[0]),
        extra={
            "boundaries": boundaries.tolist(),
            "information_fractions": fractions.tolist(),
            "power": power,
        },
    )


futl = futility_boundary


def cheatsheet() -> str:
    return "futility_boundary({}) -> Futility stopping boundary."
