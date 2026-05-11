# morie.fn — function file (hadesllm/morie)
"""Total fertility rate from age-specific fertility rates."""

import numpy as np

from ._containers import ESRes


def fertility_rate(
    age_specific_rates: list | np.ndarray,
    interval: int = 5,
) -> ESRes:
    """Compute total fertility rate from age-specific fertility rates.

    .. math::

        TFR = \\sum_i ASFR_i \\times n_i

    Parameters
    ----------
    age_specific_rates : array-like
        ASFR for each age group (births per woman per year).
    interval : int
        Width of each age group in years (default 5).

    Returns
    -------
    ESRes
    """
    asfr = np.asarray(age_specific_rates, dtype=float)
    if len(asfr) == 0:
        raise ValueError("No rates provided")

    tfr = float(np.sum(asfr) * interval)
    replacement = tfr >= 2.1

    return ESRes(
        measure="TFR",
        estimate=tfr,
        extra={"n_age_groups": len(asfr), "interval": interval, "above_replacement": replacement},
    )


fertl = fertility_rate


def cheatsheet() -> str:
    return "fertility_rate({}) -> Total fertility rate from age-specific fertility rates."
