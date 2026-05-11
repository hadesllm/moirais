# morie.fn — function file (hadesllm/morie)
"""Lead-service-line risk screen from building construction year."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult

# US lead-service-line historical regulatory milestones:
#   pre-1900: lead pipes common; no regulation
#   1900-1940: peak lead-pipe installation era
#   1940-1960: transition toward copper + galvanized
#   1960-1986: copper dominant; local lead bans patchy
#   1986: Safe Drinking Water Act Amendments BAN on lead solder,
#         lead-free pipe requirement in federal law
#   2011: Reduction of Lead in Drinking Water Act — tightened to
#         ≤ 0.25% weighted lead content for "lead-free"
#
# Canadian context (Ontario):
#   pre-1950: lead service lines common in Toronto, Hamilton, London
#   1950-1975: transition period
#   1975-2000: local bans
#   2007+: federal and provincial remediation programs
#
# The function returns a risk category based on year of construction.
# This is SCREENING ONLY — actual pipe material must be confirmed
# by in-home inspection or municipal service-line database.
_US_BANDS: list[tuple[int, int, str, float]] = [
    # (min_year, max_year, label, risk_probability)
    (0,    1900, "very_high", 0.90),
    (1901, 1940, "very_high", 0.85),
    (1941, 1960, "high",      0.60),
    (1961, 1985, "moderate",  0.25),
    (1986, 2010, "low",       0.05),
    (2011, 2100, "very_low",  0.01),
]

_CANADA_BANDS: list[tuple[int, int, str, float]] = [
    (0,    1900, "very_high", 0.85),
    (1901, 1950, "very_high", 0.80),
    (1951, 1975, "high",      0.45),
    (1976, 1999, "moderate",  0.15),
    (2000, 2100, "low",       0.03),
]


def lead_service_line_risk(
    construction_year: int | np.ndarray | pd.Series,
    *,
    country: str = "us",
    with_intervals: bool = False,
) -> DescriptiveResult:
    """Screen for lead-service-line risk by building construction year.

    Year-based heuristic for plumbing lead risk, calibrated to US and
    Canadian regulatory milestones. Returns per-building probability
    and a categorical risk band. **This is screening only — confirm
    with in-home inspection or municipal service-line records before
    using for health intervention.**

    US milestones encoded:

    - 1986 SDWA Amendments: banned lead solder, required "lead-free"
      pipes.
    - 2011 Reduction of Lead in Drinking Water Act: tightened
      "lead-free" to ≤ 0.25% weighted lead content.

    Canadian milestones:

    - Pre-1950 Ontario: lead service lines common in Toronto/Hamilton.
    - 1975–1999: local bans; patchy replacement.
    - 2007+: federal remediation pressure.

    Parameters
    ----------
    construction_year : int or array-like
        Year(s) of building / service-line installation.
    country : {"us", "ca"}, default "us"
        Which regulatory timeline to apply.
    with_intervals : bool, default False
        If True, include the band boundary years and source notes
        in extra.

    Returns
    -------
    DescriptiveResult
        value = mean lead-pipe probability across inputs.
        extra has per-building label + probability, distribution by
        band, and citation.

    Examples
    --------
    A 1920 Detroit rowhouse: very-high risk

    >>> r = lead_service_line_risk(1920, country="us")
    >>> r.extra["label"]
    'very_high'
    >>> r.value
    0.85

    A 2015 Toronto new-build: low

    >>> r = lead_service_line_risk(2015, country="ca")
    >>> r.extra["label"]
    'low'

    References
    ----------
    EPA (2024). Lead and Copper Rule Revisions (LCRR) and LCR
    Improvements (LCRI).
    https://www.epa.gov/ground-water-and-drinking-water/lead-and-copper-rule

    Health Canada (2022). Guidelines for Canadian Drinking Water
    Quality — Guideline Technical Document: Lead.

    Notes
    -----
    Quote: "The pipes below remember what the pipes above forgot."

    This function is deliberately simple because the real answer
    depends on data we don't have here (utility service-line
    inventory, actual tap sampling). A year-based screen is a
    legitimate first pass for triaging *where* to look.
    """
    c = country.lower().strip()
    if c in ("us", "usa", "united states"):
        bands = _US_BANDS
    elif c in ("ca", "can", "canada"):
        bands = _CANADA_BANDS
    else:
        raise ValueError(f"country must be 'us' or 'ca', got {country!r}.")

    years = np.atleast_1d(np.asarray(construction_year, dtype=int))

    labels: list[str] = []
    probs = np.empty(years.size, dtype=float)
    for i, y in enumerate(years.ravel()):
        y_int = int(y)
        hit = False
        for ymin, ymax, lbl, p in bands:
            if ymin <= y_int <= ymax:
                labels.append(lbl)
                probs[i] = p
                hit = True
                break
        if not hit:
            raise ValueError(f"Year {y_int} not covered by any band.")

    # Tally the band distribution
    distribution: dict[str, int] = {}
    for lbl in labels:
        distribution[lbl] = distribution.get(lbl, 0) + 1

    val = float(probs.mean()) if probs.size > 1 else float(probs.item())

    extra: dict[str, object] = {
        "probability": probs.tolist() if probs.size > 1 else float(probs.item()),
        "label": labels if len(labels) > 1 else labels[0],
        "band_distribution": distribution,
        "country": c,
        "mean_probability": val,
        "source": ("EPA LCRR/LCRI 2024" if c.startswith("u")
                    else "Health Canada 2022 Guideline Lead"),
    }
    if with_intervals:
        extra["bands"] = bands

    return DescriptiveResult(name="lead_service_line_risk", value=val, extra=extra)


leadsl = lead_service_line_risk


def cheatsheet() -> str:
    return "leadsl(year, country='us') -> lead-pipe probability by era."
