"""Vaccine efficacy from clinical trial data."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def vaccine_efficacy(
    ar_vaccinated: float,
    ar_placebo: float,
    n_vaccinated: int = 0,
    n_placebo: int = 0,
    confidence: float = 0.95,
) -> ESRes:
    """Compute vaccine efficacy (VE = 1 - RR).

    Parameters
    ----------
    ar_vaccinated : float
        Attack rate in vaccinated group.
    ar_placebo : float
        Attack rate in placebo group.
    n_vaccinated, n_placebo : int
        Sample sizes (for CI).
    confidence : float

    Returns
    -------
    ESRes
    """
    if ar_placebo <= 0:
        raise ValueError("ar_placebo must be positive")

    rr = ar_vaccinated / ar_placebo
    ve = (1 - rr) * 100

    ci_lo, ci_hi = None, None
    if n_vaccinated > 0 and n_placebo > 0:
        se_log_rr = np.sqrt(
            (1 - ar_vaccinated) / (ar_vaccinated * n_vaccinated + 1e-10)
            + (1 - ar_placebo) / (ar_placebo * n_placebo + 1e-10)
        )
        z = stats.norm.ppf((1 + confidence) / 2)
        rr_lo = rr * np.exp(-z * se_log_rr)
        rr_hi = rr * np.exp(z * se_log_rr)
        ci_lo = float((1 - rr_hi) * 100)
        ci_hi = float((1 - rr_lo) * 100)

    return ESRes(
        measure="vaccine_efficacy_pct",
        estimate=float(ve),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        extra={"relative_risk": float(rr)},
    )


vacef = vaccine_efficacy


def cheatsheet() -> str:
    return "vaccine_efficacy({}) -> Vaccine efficacy from clinical trial data."
