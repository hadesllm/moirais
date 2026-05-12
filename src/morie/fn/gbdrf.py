# morie.fn -- function file (hadesllm/morie)
"""GBD risk factor attribution (population attributable fraction)."""


from ._containers import DescriptiveResult


def gbd_risk_factor(
    rr: float,
    prevalence: float,
    n_conditions: int = 1,
) -> DescriptiveResult:
    """Compute population attributable fraction for a risk factor.

    .. math::

        PAF = \\frac{p(RR - 1)}{p(RR - 1) + 1}

    Parameters
    ----------
    rr : float
        Relative risk.
    prevalence : float
        Prevalence of risk factor (0-1).
    n_conditions : int
        Number of conditions attributable.

    Returns
    -------
    DescriptiveResult
    """
    if rr < 0 or prevalence < 0 or prevalence > 1:
        raise ValueError("Invalid rr or prevalence")

    paf = prevalence * (rr - 1) / (prevalence * (rr - 1) + 1)

    return DescriptiveResult(
        name="gbd_risk_factor",
        value=float(paf),
        extra={
            "rr": float(rr),
            "prevalence": float(prevalence),
            "n_conditions": n_conditions,
            "paf_pct": float(paf * 100),
        },
    )


gbdrf = gbd_risk_factor


def cheatsheet() -> str:
    return "gbd_risk_factor({}) -> GBD risk factor attribution (population attributable fractio"
