"""Sample size for non-inferiority trial."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def sample_size_noninferiority(
    delta: float,
    margin: float,
    sd: float,
    *,
    alpha: float = 0.025,
    power: float = 0.80,
) -> ESRes:
    """
    Sample size for a non-inferiority trial (one-sided).

    Parameters
    ----------
    delta : float
        Expected true difference (treatment - control).
    margin : float
        Non-inferiority margin (positive).
    sd : float
        Common standard deviation.
    alpha : float
        One-sided significance level (default 0.025).
    power : float
        Desired power.

    Returns
    -------
    ESRes

    References
    ----------
    D'Agostino, R. B., et al. (2003). Non-inferiority trials: design
    concepts and issues. *Stat Med*, 22(2), 169-186.
    """
    if margin <= 0 or sd <= 0:
        raise ValueError("margin and sd must be positive.")

    z_a = stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    n = ((z_a + z_b) * sd / (delta + margin)) ** 2
    n = int(np.ceil(n))

    return ESRes(
        measure="sample_size_noninferiority",
        estimate=float(n),
        extra={"n_per_group": n, "total": 2 * n, "margin": margin, "delta": delta},
    )


ssin = sample_size_noninferiority


def cheatsheet() -> str:
    return "sample_size_noninferiority({}) -> Sample size for non-inferiority trial."
