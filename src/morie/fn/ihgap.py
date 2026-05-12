# morie.fn -- function file (hadesllm/morie)
"""Health gap indicator (Indigenous vs general population)."""

from ._containers import ESRes


def health_gap(
    rate_indigenous: float,
    rate_general: float,
) -> ESRes:
    """Compute absolute and relative health gap.

    Parameters
    ----------
    rate_indigenous : float
    rate_general : float

    Returns
    -------
    ESRes
    """
    if rate_general < 0 or rate_indigenous < 0:
        raise ValueError("Rates must be non-negative")

    absolute_gap = rate_indigenous - rate_general
    relative_gap = (rate_indigenous / rate_general - 1) * 100 if rate_general > 0 else float("inf")

    return ESRes(
        measure="health_gap",
        estimate=float(absolute_gap),
        extra={
            "relative_gap_pct": float(relative_gap),
            "rate_indigenous": float(rate_indigenous),
            "rate_general": float(rate_general),
            "ratio": float(rate_indigenous / rate_general) if rate_general > 0 else float("inf"),
        },
    )


ihgap = health_gap


def cheatsheet() -> str:
    return "health_gap({}) -> Health gap indicator (Indigenous vs general population)."
