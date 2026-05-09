# moirais.fn — function file (hadesllm/moirais)
"""Racial disparity index (RDI)."""

from __future__ import annotations

from moirais.fn._containers import ESRes


def racial_disparity_index(
    rates_by_group: dict[str, float],
    *,
    reference_group: str | None = None,
) -> ESRes:
    """Compute racial disparity index (rate ratio relative to reference).

    Parameters
    ----------
    rates_by_group : dict
        Group name -> rate.
    reference_group : str, optional
        Reference group for ratio. Defaults to group with lowest rate.

    Returns
    -------
    ESRes
    """
    if len(rates_by_group) < 2:
        raise ValueError("Need at least 2 groups")
    if reference_group is None:
        reference_group = min(rates_by_group, key=rates_by_group.get)
    ref_rate = rates_by_group[reference_group]
    if ref_rate <= 0:
        raise ValueError("Reference group rate must be positive")
    ratios = {k: v / ref_rate for k, v in rates_by_group.items()}
    max_ratio = max(ratios.values())
    return ESRes(
        measure="racial_disparity_index",
        estimate=max_ratio,
        n=len(rates_by_group),
        extra={"ratios": ratios, "reference_group": reference_group, "reference_rate": ref_rate},
    )


eqrid = racial_disparity_index


def cheatsheet() -> str:
    return "racial_disparity_index({}) -> Racial disparity index (RDI)."
