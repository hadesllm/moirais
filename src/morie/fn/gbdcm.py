# morie.fn -- function file (hadesllm/morie)
"""Compare burden across conditions (GBD-style)."""


from ._containers import DescriptiveResult


def gbd_compare(
    daly_by_condition: dict[str, float],
) -> DescriptiveResult:
    """Rank and compare disease burden across conditions.

    Parameters
    ----------
    daly_by_condition : dict
        {condition_name: DALYs}.

    Returns
    -------
    DescriptiveResult
    """
    if not daly_by_condition:
        raise ValueError("No conditions provided")

    total = sum(daly_by_condition.values())
    ranked = sorted(daly_by_condition.items(), key=lambda x: x[1], reverse=True)
    proportions = {k: v / total * 100 if total > 0 else 0 for k, v in ranked}

    return DescriptiveResult(
        name="gbd_compare",
        value=proportions,
        extra={
            "total_dalys": float(total),
            "top_condition": ranked[0][0],
            "top_pct": float(ranked[0][1] / total * 100) if total > 0 else 0,
            "n_conditions": len(daly_by_condition),
        },
    )


gbdcm = gbd_compare


def cheatsheet() -> str:
    return "gbd_compare({}) -> Compare burden across conditions (GBD-style)."
