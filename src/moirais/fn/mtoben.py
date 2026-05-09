# moirais.fn — function file (hadesllm/moirais)
"""Road safety intervention benefit-cost ratio."""

from __future__ import annotations

from moirais.fn._containers import ESRes


def mto_benefit_cost(
    crash_reduction: int,
    cost: float,
    *,
    value_per_crash: float = 200000.0,
) -> ESRes:
    """Compute benefit-cost ratio for road safety intervention.

    Parameters
    ----------
    crash_reduction : int
        Number of crashes avoided.
    cost : float
        Total intervention cost.
    value_per_crash : float
        Social cost per crash avoided.

    Returns
    -------
    ESRes
    """
    if cost <= 0:
        raise ValueError("cost must be positive")
    benefit = crash_reduction * value_per_crash
    bcr = benefit / cost
    return ESRes(
        measure="road_safety_bcr", estimate=bcr, extra={"benefit": benefit, "cost": cost, "net": benefit - cost}
    )


mtoben = mto_benefit_cost


def cheatsheet() -> str:
    return "mto_benefit_cost({}) -> Road safety intervention benefit-cost ratio."
