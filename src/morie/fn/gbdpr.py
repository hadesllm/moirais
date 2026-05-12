# morie.fn -- function file (hadesllm/morie)
"""Project future disease burden."""


from ._containers import DescriptiveResult


def gbd_projection(
    current_burden: float,
    growth_rate: float,
    years: int,
) -> DescriptiveResult:
    """Project future disease burden assuming constant growth rate.

    Parameters
    ----------
    current_burden : float
    growth_rate : float
        Annual growth rate (e.g. 0.02 for 2%).
    years : int
        Projection horizon.

    Returns
    -------
    DescriptiveResult
    """
    if years < 0:
        raise ValueError("years must be non-negative")

    projected = [current_burden * (1 + growth_rate) ** t for t in range(years + 1)]

    return DescriptiveResult(
        name="gbd_projection",
        value=float(projected[-1]),
        extra={
            "current": float(current_burden),
            "growth_rate": float(growth_rate),
            "years": years,
            "trajectory": projected,
            "total_cumulative": float(sum(projected)),
        },
    )


gbdpr = gbd_projection


def cheatsheet() -> str:
    return "gbd_projection({}) -> Project future disease burden."
