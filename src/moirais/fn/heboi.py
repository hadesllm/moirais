# moirais.fn — function file (hadesllm/moirais)
"""Burden of illness (total societal)."""

from ._containers import ESRes


def burden_of_illness(
    direct_costs: float,
    indirect_costs: float,
    intangible_costs: float = 0.0,
) -> ESRes:
    """Compute total burden of illness.

    Parameters
    ----------
    direct_costs : float
    indirect_costs : float
    intangible_costs : float

    Returns
    -------
    ESRes
    """
    total = direct_costs + indirect_costs + intangible_costs
    return ESRes(
        measure="burden_of_illness",
        estimate=float(total),
        extra={
            "direct": float(direct_costs),
            "indirect": float(indirect_costs),
            "intangible": float(intangible_costs),
            "pct_direct": float(direct_costs / total * 100) if total > 0 else 0.0,
        },
    )


heboi = burden_of_illness


def cheatsheet() -> str:
    return "burden_of_illness({}) -> Burden of illness (total societal)."
