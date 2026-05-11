"""Herd immunity threshold for vaccination."""

from ._containers import ESRes


def vaccine_herd(
    R0: float,
    vaccine_efficacy: float = 1.0,
) -> ESRes:
    """Compute herd immunity threshold accounting for vaccine efficacy.

    .. math::

        p_c = \\frac{1 - 1/R_0}{VE}

    Parameters
    ----------
    R0 : float
        Basic reproduction number.
    vaccine_efficacy : float
        Vaccine efficacy (0-1).

    Returns
    -------
    ESRes
    """
    if R0 <= 1:
        return ESRes(
            measure="herd_immunity_threshold",
            estimate=0.0,
            extra={"R0": float(R0), "note": "R0 <= 1, no herd threshold needed"},
        )
    if vaccine_efficacy <= 0 or vaccine_efficacy > 1:
        raise ValueError("vaccine_efficacy must be in (0, 1]")

    critical_vacc = (1 - 1 / R0) / vaccine_efficacy

    return ESRes(
        measure="herd_immunity_threshold",
        estimate=float(min(critical_vacc, 1.0)),
        extra={
            "R0": float(R0),
            "vaccine_efficacy": float(vaccine_efficacy),
            "achievable": critical_vacc <= 1.0,
            "pct_needed": float(min(critical_vacc, 1.0) * 100),
        },
    )


vachrd = vaccine_herd


def cheatsheet() -> str:
    return "vaccine_herd({}) -> Herd immunity threshold for vaccination."
