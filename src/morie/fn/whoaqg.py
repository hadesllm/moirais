"""WHO 2021 Air Quality Guidelines compliance check."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._containers import DescriptiveResult

# WHO Global Air Quality Guidelines 2021, 2nd ed.
# https://www.who.int/publications/i/item/9789240034228
# Annual mean (AM) and 24-hour mean (24h) thresholds, µg/m³ unless noted.
#
# Plus four interim targets (IT-1 through IT-4) per pollutant, which
# many countries still aim at.
_WHO_AQG_2021: dict[str, dict[str, Any]] = {
    "pm25": {
        "annual":   {"aqg": 5,  "it1": 35, "it2": 25, "it3": 15, "it4": 10},
        "24h":      {"aqg": 15, "it1": 75, "it2": 50, "it3": 37.5, "it4": 25},
        "unit":     "µg/m³",
    },
    "pm10": {
        "annual":   {"aqg": 15, "it1": 70, "it2": 50, "it3": 30, "it4": 20},
        "24h":      {"aqg": 45, "it1": 150, "it2": 100, "it3": 75, "it4": 50},
        "unit":     "µg/m³",
    },
    "no2": {
        "annual":   {"aqg": 10, "it1": 40, "it2": 30, "it3": 20},
        "24h":      {"aqg": 25, "it1": 120, "it2": 50},
        "unit":     "µg/m³",
    },
    "o3": {
        "peak_season": {"aqg": 60, "it1": 100, "it2": 70},
        "8h":          {"aqg": 100, "it1": 160, "it2": 120},
        "unit":     "µg/m³",
    },
    "so2": {
        "24h":      {"aqg": 40, "it1": 125, "it2": 50},
        "unit":     "µg/m³",
    },
    "co": {
        "24h":      {"aqg": 4,  "it1": 7},
        "unit":     "mg/m³",
    },
}


def who_aqg_compliance(
    concentration: float | np.ndarray,
    pollutant: str,
    averaging: str,
) -> DescriptiveResult:
    """Check concentration against WHO 2021 Air Quality Guidelines.

    Reports compliance level using the WHO guideline (AQG) and the
    four interim targets (IT-1, IT-2, IT-3, IT-4). A level of "AQG"
    means the measurement meets the full WHO recommended value;
    "IT-1" is the most permissive interim target (often the current
    bar for high-pollution regions); "fail" means worse than IT-1.

    Parameters
    ----------
    concentration : float or array-like
        Measured concentration (µg/m³ for all pollutants except CO
        which is mg/m³). For averaging="annual" use annual mean; for
        "24h" use 24-hour mean.
    pollutant : {"pm25", "pm10", "no2", "o3", "so2", "co"}
        Pollutant identifier.
    averaging : str
        Averaging window: "annual", "24h", "8h" (O₃ only), or
        "peak_season" (O₃ only). See WHO 2021 Table A.2.1.

    Returns
    -------
    DescriptiveResult
        value = fraction of observations meeting the WHO AQG (0.0-1.0
        for arrays, 0.0 or 1.0 for scalar).
        extra has per-observation level (AQG/IT-4/.../fail), the
        thresholds used, n_obs, and a summary compliance rate.

    Examples
    --------
    Annual PM₂.₅ of 12 µg/m³ (above WHO AQG of 5):

    >>> r = who_aqg_compliance(12, "pm25", "annual")
    >>> r.extra["level"]
    'IT-4'

    References
    ----------
    WHO (2021). WHO Global Air Quality Guidelines. Particulate Matter
    (PM₂.₅ and PM₁₀), Ozone, Nitrogen Dioxide, Sulfur Dioxide and
    Carbon Monoxide. World Health Organization.
    """
    p = pollutant.lower().strip()
    # Accept "peak-season", "peak_season", "peak season" as equivalent
    a = averaging.lower().strip().replace("-", "_").replace(" ", "_")

    if p not in _WHO_AQG_2021:
        raise KeyError(
            f"Unknown pollutant {pollutant!r}. Available: {list(_WHO_AQG_2021)}"
        )
    bands = _WHO_AQG_2021[p]
    if a not in bands:
        valid = [k for k in bands if k != "unit"]
        raise KeyError(
            f"Unknown averaging {averaging!r} for {p}. Available: {valid}"
        )
    thresholds = bands[a]
    # Order from strictest (aqg) to most permissive (it1). Missing
    # tiers are fine — e.g. SO₂ has only AQG + IT-1 + IT-2.
    ordered_tiers = [t for t in ("aqg", "it4", "it3", "it2", "it1")
                     if t in thresholds]

    C = np.atleast_1d(np.asarray(concentration, dtype=float))
    if np.any(C < 0):
        raise ValueError("Concentrations must be non-negative.")

    def _level(c: float) -> str:
        for tier in ordered_tiers:
            if c <= thresholds[tier]:
                return tier.upper() if tier == "aqg" else tier.upper().replace("IT", "IT-")
        return "fail"

    levels = [_level(float(c)) for c in C.ravel()]
    meets_aqg = np.array([lvl == "AQG" for lvl in levels], dtype=float)
    compliance_rate = float(meets_aqg.mean())

    return DescriptiveResult(
        name="who_aqg_compliance",
        value=compliance_rate,
        extra={
            "level": levels if C.size > 1 else levels[0],
            "compliance_rate": compliance_rate,
            "thresholds": thresholds,
            "pollutant": p,
            "averaging": a,
            "unit": bands["unit"],
            "n_obs": int(C.size),
            "source": "WHO 2021 Global Air Quality Guidelines",
        },
    )


whoaqg = who_aqg_compliance


def cheatsheet() -> str:
    return "whoaqg(C, pollutant, averaging) -> WHO AQG/IT compliance band."
