# moirais.fn — function file (hadesllm/moirais)
"""PFAS Hazard Index per US EPA 2024 MCL methodology."""

from __future__ import annotations

from ._containers import DescriptiveResult

# US EPA 2024 final PFAS National Primary Drinking Water Regulation
# (April 2024). Individual MCLs for PFOA and PFOS (4 parts per trillion
# each), plus a Hazard Index approach for the 4-compound mixture.
# https://www.epa.gov/sdwa/and-polyfluoroalkyl-substances-pfas
#
# Hazard Quotient (HQ) for compound i = C_i / HBWC_i
# where HBWC is the Health-Based Water Concentration. For regulated
# non-PFOA/PFOS compounds, HBWC values in ppt (ng/L):
#
#   PFHxS:     10 ppt
#   HFPO-DA:   10 ppt  (GenX chemicals — DuPont/Chemours Cape Fear)
#   PFNA:      10 ppt
#   PFBS:    2000 ppt
#
# Hazard Index = sum of HQs across the four compounds. HI > 1 is the
# enforceable violation threshold.
_EPA_HBWC_PPT: dict[str, float] = {
    "pfhxs":   10.0,
    "hfpo-da": 10.0,   # Also called GenX
    "genx":    10.0,   # alias
    "pfna":    10.0,
    "pfbs":  2000.0,
}

# Individual MCLs (ppt) for PFOA/PFOS — these are *not* part of the
# Hazard Index; they have their own single-compound thresholds.
_EPA_MCL_PPT: dict[str, float] = {
    "pfoa": 4.0,
    "pfos": 4.0,
}


def pfas_hazard_index(
    concentrations_ppt: dict[str, float],
) -> DescriptiveResult:
    """Compute EPA 2024 PFAS Hazard Index + individual PFOA/PFOS compliance.

    Evaluates a drinking-water PFAS measurement against the US EPA's
    April 2024 National Primary Drinking Water Regulation. Two regimes:

    1. **Individual MCLs** (4 ppt each) for PFOA and PFOS — hard caps.
    2. **Hazard Index** for the mixture of PFHxS + HFPO-DA + PFNA + PFBS.
       HI = Σ (C_i / HBWC_i). HI > 1 is a regulatory violation.

    Parameters
    ----------
    concentrations_ppt : dict[str, float]
        Measured concentrations in parts-per-trillion (ng/L), keyed
        by compound name (case-insensitive). Recognized names:
        ``pfoa``, ``pfos``, ``pfhxs``, ``hfpo-da`` (a.k.a. ``genx``),
        ``pfna``, ``pfbs``.

        Compounds not included default to 0 for the HI sum. Unknown
        names raise KeyError with the available-names list.

    Returns
    -------
    DescriptiveResult
        value = Hazard Index (unitless; > 1 is a violation).
        extra contains:

        - ``pfoa_compliance``, ``pfos_compliance`` — bool, True if ≤ 4 ppt
        - ``hazard_index`` — float, the mixture HI
        - ``per_compound_hq`` — dict of HQs for the 4 HI compounds
        - ``violation`` — list of reasons for non-compliance (empty if clean)
        - ``epa_thresholds`` — the numeric bounds used, for reference

    Examples
    --------
    A municipal water sample with low PFOA but elevated PFHxS:

    >>> r = pfas_hazard_index({
    ...     "pfoa":    2.0,    # ≤ 4 ppt — compliant
    ...     "pfos":    1.5,    # ≤ 4 ppt — compliant
    ...     "pfhxs":   8.0,    # HQ = 0.8
    ...     "hfpo-da": 5.0,    # HQ = 0.5
    ... })
    >>> round(r.value, 2)      # HI = 0.8 + 0.5 = 1.3 → violation
    1.3
    >>> "pfhxs+hfpo-da HI" in r.extra["violation"][0]
    True

    References
    ----------
    US EPA (2024). PFAS National Primary Drinking Water Regulation.
    Final rule, 40 CFR Parts 141 and 142. Federal Register 89(76),
    32532-32757, April 2024.
    https://www.epa.gov/sdwa/and-polyfluoroalkyl-substances-pfas

    Notes
    -----
    Quote: "The forever chemicals do not forget; they accumulate."
    """
    # Normalize keys
    conc = {k.lower().strip(): float(v) for k, v in concentrations_ppt.items()}
    for c in conc:
        if c not in _EPA_HBWC_PPT and c not in _EPA_MCL_PPT:
            raise KeyError(
                f"Unknown PFAS compound {c!r}. Recognized: "
                f"{sorted(set(_EPA_HBWC_PPT) | set(_EPA_MCL_PPT))}"
            )
    if any(v < 0 for v in conc.values()):
        raise ValueError("Concentrations must be non-negative.")

    # Individual PFOA/PFOS compliance
    pfoa_c = conc.get("pfoa", 0.0)
    pfos_c = conc.get("pfos", 0.0)
    pfoa_ok = pfoa_c <= _EPA_MCL_PPT["pfoa"]
    pfos_ok = pfos_c <= _EPA_MCL_PPT["pfos"]

    # Hazard Index (dedupe GenX alias)
    hq: dict[str, float] = {}
    hi = 0.0
    for cmp in ("pfhxs", "hfpo-da", "pfna", "pfbs"):
        # Treat 'genx' as synonym for 'hfpo-da'
        val = conc.get(cmp, conc.get("genx", 0.0) if cmp == "hfpo-da" else 0.0)
        q = val / _EPA_HBWC_PPT[cmp]
        hq[cmp] = q
        hi += q

    violations = []
    if not pfoa_ok:
        violations.append(f"PFOA {pfoa_c} ppt > MCL {_EPA_MCL_PPT['pfoa']} ppt")
    if not pfos_ok:
        violations.append(f"PFOS {pfos_c} ppt > MCL {_EPA_MCL_PPT['pfos']} ppt")
    if hi > 1.0:
        violations.append(f"pfhxs+hfpo-da+pfna+pfbs HI = {hi:.3f} > 1.0")

    return DescriptiveResult(
        name="pfas_hazard_index",
        value=float(hi),
        extra={
            "hazard_index": float(hi),
            "per_compound_hq": {k: float(v) for k, v in hq.items()},
            "pfoa_ppt": pfoa_c,
            "pfos_ppt": pfos_c,
            "pfoa_compliance": pfoa_ok,
            "pfos_compliance": pfos_ok,
            "violation": violations,
            "compliant": (pfoa_ok and pfos_ok and hi <= 1.0),
            "epa_thresholds": {
                "mcl_pfoa_ppt": _EPA_MCL_PPT["pfoa"],
                "mcl_pfos_ppt": _EPA_MCL_PPT["pfos"],
                "hbwc_ppt": dict(_EPA_HBWC_PPT),
                "hi_threshold": 1.0,
            },
            "source": "US EPA 2024 PFAS NPDWR (40 CFR 141, 142)",
        },
    )


pfashi = pfas_hazard_index


def cheatsheet() -> str:
    return "pfashi({'pfoa':x, 'pfhxs':y, ...}) -> EPA 2024 PFAS Hazard Index."
