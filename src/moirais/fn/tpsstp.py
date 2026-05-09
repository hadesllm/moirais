"""Stop and search disparity analysis."""

from __future__ import annotations

from moirais.fn._containers import DescriptiveResult


def tps_stop_and_search(
    stops: dict[str, int],
    population: dict[str, int],
) -> DescriptiveResult:
    """Compute stop-and-search disparity ratios by group.

    For each demographic group, computes the rate ratio relative
    to the overall population stop rate.

    Parameters
    ----------
    stops : dict
        Group name -> number of stops.
    population : dict
        Group name -> population count.

    Returns
    -------
    DescriptiveResult
    """
    total_stops = sum(stops.values())
    total_pop = sum(population.values())
    if total_pop <= 0:
        raise ValueError("Total population must be positive")
    overall_rate = total_stops / total_pop
    ratios = {}
    rates = {}
    for grp in stops:
        if grp not in population or population[grp] <= 0:
            continue
        grp_rate = stops[grp] / population[grp]
        rates[grp] = grp_rate
        ratios[grp] = grp_rate / overall_rate if overall_rate > 0 else float("nan")
    return DescriptiveResult(
        name="stop_and_search_disparity",
        value=float(max(ratios.values())) if ratios else 0.0,
        extra={
            "rates": rates,
            "disparity_ratios": ratios,
            "overall_rate": overall_rate,
            "total_stops": total_stops,
        },
    )


tpsstp = tps_stop_and_search


def cheatsheet() -> str:
    return "tps_stop_and_search({}) -> Stop and search disparity analysis."
