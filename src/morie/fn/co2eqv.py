# morie.fn — function file (hadesllm/morie)
"""CO₂-equivalent conversion using IPCC AR6 GWP100 factors."""

from __future__ import annotations

from ._containers import DescriptiveResult

# IPCC AR6 Working Group I (2021) — Chapter 7, Table 7.SM.7
# GWP-100 values (excluding climate-carbon feedbacks). Units: kg CO₂-eq / kg gas.
_GWP100_AR6: dict[str, float] = {
    "co2": 1.0,
    "ch4": 27.9,          # Methane, fossil origin
    "ch4_bio": 27.0,      # Methane, biogenic origin
    "n2o": 273.0,         # Nitrous oxide
    "sf6": 25200.0,       # Sulfur hexafluoride
    "nf3": 17400.0,       # Nitrogen trifluoride
    "cf4": 7380.0,        # Tetrafluoromethane (PFC-14)
    "c2f6": 12400.0,      # Hexafluoroethane (PFC-116)
    "hfc-134a": 1530.0,   # Most common mobile-AC refrigerant
    "hfc-23":  14600.0,   # Byproduct of HCFC-22 manufacture
    "hfc-32":  771.0,     # R32, modern stationary AC
    "hfc-125": 3740.0,    # R125, blend component
    "hfc-143a": 5810.0,   # R143a, blend component
    "hfc-152a": 164.0,    # HFC-152a
    "hfc-227ea": 3600.0,  # Medical aerosol propellant
    "hfc-236fa": 8690.0,
    "hfc-245fa": 962.0,
    "hfc-404a": 3940.0,   # Common commercial refrigerant blend
    "hfc-407c": 1620.0,
    "hfc-410a": 1920.0,   # Residential AC blend
    "hfc-507a": 3990.0,
}


def co2_equivalent(
    emissions_kg: dict[str, float] | float,
    gas: str | None = None,
) -> DescriptiveResult:
    """Difficulties strengthen the mind, as labor does the body. — Seneca"""
    if isinstance(emissions_kg, dict):
        total = 0.0
        breakdown: dict[str, float] = {}
        factors_used: dict[str, float] = {}
        for g, amt in emissions_kg.items():
            key = g.lower().strip()
            if key not in _GWP100_AR6:
                raise KeyError(
                    f"Unknown gas {g!r}. Available: {sorted(_GWP100_AR6)}"
                )
            gwp = _GWP100_AR6[key]
            contrib = float(amt) * gwp
            total += contrib
            breakdown[key] = contrib
            factors_used[key] = gwp
        return DescriptiveResult(
            name="co2_equivalent",
            value=float(total),
            extra={
                "co2eq_kg": float(total),
                "breakdown_kg": breakdown,
                "gwp100_factors": factors_used,
                "source": "IPCC AR6 WGI Table 7.SM.7",
            },
        )

    # Scalar form
    if gas is None:
        raise ValueError("Scalar emissions_kg requires gas= argument.")
    key = gas.lower().strip()
    if key not in _GWP100_AR6:
        raise KeyError(
            f"Unknown gas {gas!r}. Available: {sorted(_GWP100_AR6)}"
        )
    gwp = _GWP100_AR6[key]
    total = float(emissions_kg) * gwp
    return DescriptiveResult(
        name="co2_equivalent",
        value=float(total),
        extra={
            "co2eq_kg": float(total),
            "gas": key,
            "gwp100": gwp,
            "source": "IPCC AR6 WGI Table 7.SM.7",
        },
    )


co2eqv = co2_equivalent


def cheatsheet() -> str:
    return "co2eqv({gas: kg} | kg, gas=...) -> CO₂-eq kg (IPCC AR6 GWP100)."
