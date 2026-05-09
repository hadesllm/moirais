"""Cost-benefit analysis of policing intervention."""

from __future__ import annotations

from moirais.fn._containers import ESRes


def tps_cost_benefit(
    costs: float,
    benefits: float,
    *,
    discount_rate: float = 0.03,
    years: int = 1,
) -> ESRes:
    """Compute benefit-cost ratio for policing intervention.

    Parameters
    ----------
    costs : float
        Total program costs.
    benefits : float
        Total estimated benefits (avoided crimes * social cost).
    discount_rate : float
        Annual discount rate for NPV.
    years : int
        Program duration in years.

    Returns
    -------
    ESRes
    """
    if costs <= 0:
        raise ValueError("costs must be positive")
    npv_benefits = sum(benefits / (1 + discount_rate) ** t for t in range(1, years + 1))
    npv_costs = sum(costs / (1 + discount_rate) ** t for t in range(1, years + 1))
    bcr = npv_benefits / npv_costs if npv_costs > 0 else float("inf")
    net = npv_benefits - npv_costs
    return ESRes(
        measure="benefit_cost_ratio",
        estimate=bcr,
        extra={
            "npv_benefits": npv_benefits,
            "npv_costs": npv_costs,
            "net_benefit": net,
            "discount_rate": discount_rate,
        },
    )


tpscb = tps_cost_benefit


def cheatsheet() -> str:
    return "tps_cost_benefit({}) -> Cost-benefit analysis of policing intervention."
