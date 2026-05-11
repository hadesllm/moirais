"""Sample size for equivalence trial."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def sample_size_equivalence(
    delta: float,
    margin: float,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> ESRes:
    """
    Sample size for a two-one-sided-tests (TOST) equivalence trial.

    Parameters
    ----------
    delta : float
        True expected difference between groups.
    margin : float
        Equivalence margin (symmetric).
    sd : float
        Common standard deviation.
    alpha : float
        One-sided significance level for each TOST test.
    power : float
        Desired power.

    Returns
    -------
    ESRes

    References
    ----------
    Schuirmann, D. J. (1987). A comparison of the two one-sided tests
    procedure and the power approach for assessing the equivalence of
    average bioavailability. *J Pharmacokinet Biopharm*, 15(6), 657-680.
    """
    if margin <= 0 or sd <= 0:
        raise ValueError("margin and sd must be positive.")
    if abs(delta) >= margin:
        raise ValueError("|delta| must be less than margin for feasibility.")

    z_a = stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    n = ((z_a + z_b) * sd / (margin - abs(delta))) ** 2
    n = int(np.ceil(n))

    return ESRes(
        measure="sample_size_equivalence",
        estimate=float(n),
        extra={"n_per_group": n, "total": 2 * n, "margin": margin, "delta": delta},
    )


ssie = sample_size_equivalence


def cheatsheet() -> str:
    return "sample_size_equivalence({}) -> Sample size for equivalence trial."
