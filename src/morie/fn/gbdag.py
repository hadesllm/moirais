# morie.fn -- function file (hadesllm/morie)
"""Age pattern of disease burden (GBD-style)."""


from ._containers import DescriptiveResult


def gbd_age_pattern(
    dalys_by_age: dict[str, float],
) -> DescriptiveResult:
    """Analyse age pattern of disease burden.

    Parameters
    ----------
    dalys_by_age : dict
        {age_group: DALYs}.

    Returns
    -------
    DescriptiveResult
    """
    if not dalys_by_age:
        raise ValueError("No data provided")

    total = sum(dalys_by_age.values())
    proportions = {k: v / total * 100 if total > 0 else 0 for k, v in dalys_by_age.items()}
    peak_group = max(dalys_by_age, key=dalys_by_age.get)

    return DescriptiveResult(
        name="gbd_age_pattern",
        value=proportions,
        extra={"total_dalys": float(total), "peak_age_group": peak_group, "n_groups": len(dalys_by_age)},
    )


gbdag = gbd_age_pattern


def cheatsheet() -> str:
    return "gbd_age_pattern({}) -> Age pattern of disease burden (GBD-style)."
