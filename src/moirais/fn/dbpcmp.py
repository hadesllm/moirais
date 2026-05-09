# moirais.fn — function file (hadesllm/moirais)
"""Disinfection-byproducts (TTHMs, HAA5) drinking-water compliance."""

from __future__ import annotations

from ._containers import DescriptiveResult

# US EPA Stage 2 Disinfection Byproducts Rule (DBPR), finalized 2006,
# effective 2012 for all water systems.
# https://www.epa.gov/dwreginfo/stage-2-disinfectants-and-disinfection-byproducts-rule
#
#   TTHMs (total trihalomethanes) MCL:  80 µg/L (LRAA)
#   HAA5  (5 haloacetic acids)   MCL:  60 µg/L (LRAA)
#   Bromate:                     10 µg/L (annual running average)
#   Chlorite:                  1.0 mg/L (monthly average)
#
# Health Canada guidelines (2020 technical documents):
#   TTHMs:  100 µg/L  (running annual average)
#   HAA5:    80 µg/L  (running annual average)
#   Bromate: 10 µg/L  (maximum)

_DBP_MCL: dict[str, dict[str, tuple[float, str]]] = {
    "us": {
        "tthm":      (80.0, "µg/L"),
        "haa5":      (60.0, "µg/L"),
        "bromate":   (10.0, "µg/L"),
        "chlorite":  (1000.0, "µg/L"),
    },
    "ca": {
        "tthm":     (100.0, "µg/L"),
        "haa5":     (80.0, "µg/L"),
        "bromate":  (10.0, "µg/L"),
    },
}


def dbp_compliance(
    concentrations: dict[str, float],
    *,
    country: str = "us",
) -> DescriptiveResult:
    """Check drinking-water disinfection-byproducts (DBP) compliance.

    Validates a set of measured DBP concentrations against US EPA
    Stage 2 DBPR (2006/2012) or Health Canada 2020 guidelines.

    Parameters
    ----------
    concentrations : dict[str, float]
        Dict keyed by DBP name (case-insensitive), values in µg/L.
        Recognized species: ``tthm`` (total trihalomethanes),
        ``haa5`` (haloacetic acids), ``bromate``, ``chlorite`` (US).
    country : {"us", "ca"}, default "us"
        Regulatory framework to apply. Canada's TTHMs/HAA5 limits
        are looser (100 / 80 vs US 80 / 60).

    Returns
    -------
    DescriptiveResult
        value = maximum (concentration / MCL) ratio across species.
        value ≤ 1.0 = compliant.
        extra has per-species ratio, MCL, units, compliance booleans,
        and a `violations` list with human-readable reasons.

    Examples
    --------
    A Canadian system with elevated TTHMs:

    >>> r = dbp_compliance({"tthm": 105.0, "haa5": 40.0}, country="ca")
    >>> r.extra["tthm_compliance"]
    False

    References
    ----------
    US EPA (2006). National Primary Drinking Water Regulations: Stage 2
    Disinfectants and Disinfection Byproducts Rule. 71 FR 388, Jan 4 2006.

    Health Canada (2020). Guidelines for Canadian Drinking Water
    Quality — Guideline Technical Document: Trihalomethanes.

    Notes
    -----
    Quote: "We chlorinate to stop cholera, and live with the
    tradeoff." — summary of drinking-water public-health history.

    Chlorine disinfection is one of the great public-health
    interventions of the 20th century. DBPs are its measurable
    tradeoff; this function quantifies the tradeoff against
    regulated limits.
    """
    c = country.lower().strip()
    if c in ("us", "usa", "united states"):
        mcl_table = _DBP_MCL["us"]
    elif c in ("ca", "can", "canada"):
        mcl_table = _DBP_MCL["ca"]
    else:
        raise ValueError(f"country must be 'us' or 'ca', got {country!r}.")

    conc = {k.lower().strip(): float(v) for k, v in concentrations.items()}
    if any(v < 0 for v in conc.values()):
        raise ValueError("Concentrations must be non-negative.")

    for k in conc:
        if k not in mcl_table:
            raise KeyError(
                f"Unknown DBP {k!r} for {c}. Recognized: {list(mcl_table)}"
            )

    ratios: dict[str, float] = {}
    per_compliance: dict[str, bool] = {}
    violations: list[str] = []
    for species, (mcl, unit) in mcl_table.items():
        if species not in conc:
            continue
        v = conc[species]
        ratio = v / mcl
        ratios[species] = ratio
        ok = ratio <= 1.0
        per_compliance[species + "_compliance"] = ok
        if not ok:
            violations.append(
                f"{species.upper()} {v:.1f} {unit} > MCL {mcl:.1f} {unit}"
            )

    max_ratio = max(ratios.values()) if ratios else 0.0
    compliant_overall = max_ratio <= 1.0

    return DescriptiveResult(
        name="dbp_compliance",
        value=float(max_ratio),
        extra={
            "ratios": ratios,
            "max_ratio": float(max_ratio),
            "compliant": compliant_overall,
            **per_compliance,
            "violations": violations,
            "country": c,
            "mcls": {k: v[0] for k, v in mcl_table.items()},
            "units": {k: v[1] for k, v in mcl_table.items()},
            "source": ("US EPA Stage 2 DBPR 2006" if c.startswith("u")
                        else "Health Canada 2020 TTHMs/HAA5"),
        },
    )


dbpcmp = dbp_compliance


def cheatsheet() -> str:
    return "dbpcmp({'tthm': x, 'haa5': y, ...}, country='us') -> DBP compliance."
