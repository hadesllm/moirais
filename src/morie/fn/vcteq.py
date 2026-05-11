"""Victimization equity across demographics."""

from __future__ import annotations

from morie.fn._containers import DescriptiveResult


def victim_equity(
    rates_by_group: dict[str, float],
    population_by_group: dict[str, int],
) -> DescriptiveResult:
    """Assess victimization equity across demographic groups.

    Computes rate ratios and identifies most/least victimized groups.

    Parameters
    ----------
    rates_by_group : dict
        Group name -> victimization rate.
    population_by_group : dict
        Group name -> population.

    Returns
    -------
    DescriptiveResult
    """
    if len(rates_by_group) < 2:
        raise ValueError("Need at least 2 groups")
    overall_rate = sum(rates_by_group[g] * population_by_group.get(g, 1) for g in rates_by_group) / sum(
        population_by_group.get(g, 1) for g in rates_by_group
    )
    ratios = {g: r / overall_rate if overall_rate > 0 else 0.0 for g, r in rates_by_group.items()}
    return DescriptiveResult(
        name="victimization_equity",
        value=float(max(ratios.values()) - min(ratios.values())),
        extra={
            "rate_ratios": ratios,
            "overall_rate": overall_rate,
            "most_victimized": max(rates_by_group, key=rates_by_group.get),
            "least_victimized": min(rates_by_group, key=rates_by_group.get),
        },
    )


vcteq = victim_equity


def cheatsheet() -> str:
    return "victim_equity({}) -> Victimization equity across demographics."
