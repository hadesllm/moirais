# morie.fn -- function file (hadesllm/morie)
"""Years lived with disability (YLD)."""

from ._containers import ESRes


def years_lived_disability(
    incidence: float,
    duration: float,
    disability_weight: float,
) -> ESRes:
    """Compute YLD = incidence x duration x disability weight.

    Parameters
    ----------
    incidence : float
    duration : float
        Average duration of condition in years.
    disability_weight : float
        GBD disability weight (0-1).

    Returns
    -------
    ESRes
    """
    if disability_weight < 0 or disability_weight > 1:
        raise ValueError("disability_weight must be in [0, 1]")
    if incidence < 0 or duration < 0:
        raise ValueError("incidence and duration must be non-negative")

    yld = incidence * duration * disability_weight

    return ESRes(
        measure="YLD",
        estimate=float(yld),
        extra={
            "incidence": float(incidence),
            "duration": float(duration),
            "disability_weight": float(disability_weight),
        },
    )


cdyld = years_lived_disability


def cheatsheet() -> str:
    return "years_lived_disability({}) -> Years lived with disability (YLD)."
