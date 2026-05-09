"""Societal cost of substance use."""

from ._containers import ESRes


def substance_cost(
    medical_costs: float,
    productivity_costs: float,
    criminal_costs: float = 0.0,
    other_costs: float = 0.0,
) -> ESRes:
    """Compute total societal cost of substance use.

    Parameters
    ----------
    medical_costs : float
        Direct medical/healthcare costs.
    productivity_costs : float
        Indirect productivity losses.
    criminal_costs : float
        Criminal justice system costs.
    other_costs : float
        Other costs (e.g. social services).

    Returns
    -------
    ESRes
    """
    total = medical_costs + productivity_costs + criminal_costs + other_costs
    if total < 0:
        raise ValueError("Total cost cannot be negative")

    return ESRes(
        measure="societal_cost",
        estimate=float(total),
        extra={
            "medical": float(medical_costs),
            "productivity": float(productivity_costs),
            "criminal": float(criminal_costs),
            "other": float(other_costs),
            "pct_medical": float(medical_costs / total * 100) if total > 0 else 0.0,
            "pct_productivity": float(productivity_costs / total * 100) if total > 0 else 0.0,
        },
    )


sucost = substance_cost


def cheatsheet() -> str:
    return "substance_cost({}) -> Societal cost of substance use."
