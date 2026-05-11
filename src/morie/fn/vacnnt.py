"""Number needed to vaccinate (NNV)."""


from ._containers import ESRes


def vaccine_nnt(
    vaccine_efficacy: float,
    baseline_risk: float,
) -> ESRes:
    """Number needed to vaccinate to prevent one case.

    .. math::

        NNV = \\frac{1}{VE \\times AR_0}

    Parameters
    ----------
    vaccine_efficacy : float
        Vaccine efficacy as proportion (0-1).
    baseline_risk : float
        Baseline attack rate (0-1).

    Returns
    -------
    ESRes
    """
    if vaccine_efficacy <= 0 or vaccine_efficacy > 1:
        raise ValueError("vaccine_efficacy must be in (0, 1]")
    if baseline_risk <= 0 or baseline_risk > 1:
        raise ValueError("baseline_risk must be in (0, 1]")

    nnv = 1 / (vaccine_efficacy * baseline_risk)

    return ESRes(
        measure="NNV",
        estimate=float(nnv),
        extra={"vaccine_efficacy": float(vaccine_efficacy), "baseline_risk": float(baseline_risk)},
    )


vacnnt = vaccine_nnt


def cheatsheet() -> str:
    return "vaccine_nnt({}) -> Number needed to vaccinate (NNV)."
