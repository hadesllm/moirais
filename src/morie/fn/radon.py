# morie.fn -- function file (hadesllm/morie)
"""Radon indoor-air cancer risk per EPA BEIR VI / ICRP methodology."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

# US EPA action level: 4 pCi/L (148 Bq/m³) triggers mitigation.
# WHO reference level: 100 Bq/m³ (2.7 pCi/L) -- stricter.
# Health Canada guideline: 200 Bq/m³ (5.4 pCi/L).
#
# Lifetime excess lung-cancer risk per pCi/L of indoor radon:
# - EPA (2003 Assessment of Risks from Radon in Homes, EPA 402-R-03-003):
#   smokers:       ~7e-3 per pCi/L (7 excess lung-cancer deaths per 1,000
#                  lifetime exposed at 1 pCi/L)
#   non-smokers:   ~0.7e-3 per pCi/L (ten-fold lower due to multiplicative
#                  interaction with tobacco)
#
# The BEIR VI committee fit excess relative risk (ERR) as linear in
# radon working-level-months (WLM):
#     ERR = 0.0117 * WLM  (ever-smokers)
#     ERR = 0.0153 * WLM  (never-smokers)
# but for indoor-air public-health estimates, the pCi/L-based lifetime
# risk is more commonly used.
_EPA_LIFETIME_RISK_PER_PCIL: dict[str, float] = {
    # Deaths per 1,000 persons over 70-year lifetime exposure
    "smoker":     7.0e-3,
    "never_smoker": 7.0e-4,
}

_BQ_M3_PER_PCIL = 37.0   # Conversion factor


def radon_cancer_risk(
    radon_level: float | np.ndarray,
    *,
    unit: str = "pCi/L",
    smoker: bool = False,
    exposure_years: float = 70.0,
) -> DescriptiveResult:
    """Estimate lifetime excess lung-cancer risk from indoor radon.

    Uses US EPA 2003 risk factors derived from BEIR VI pooled data on
    miners and residential cohorts. Returns lifetime probability of
    radon-attributable lung-cancer death for the specified exposure.

    Compliance flags follow both US EPA (action level 4 pCi/L) and
    WHO (reference level 2.7 pCi/L ≈ 100 Bq/m³).

    Parameters
    ----------
    radon_level : float or array-like
        Radon concentration.
    unit : {"pCi/L", "Bq/m3"}, default "pCi/L"
        Measurement unit. 1 pCi/L = 37 Bq/m³.
    smoker : bool, default False
        Smoking status: dramatically modifies risk (~10× higher).
        BEIR VI frames this as multiplicative interaction with radon.
    exposure_years : float, default 70
        Lifetime years of exposure. Default 70 is EPA's assumption;
        shorter residential exposure scales linearly (conservative).

    Returns
    -------
    DescriptiveResult
        value = mean lifetime excess cancer probability per person.
        extra has per-observation risk, both-unit concentration,
        EPA/WHO compliance flags, source.

    Examples
    --------
    A non-smoker household at 8 pCi/L (double EPA action level):

    >>> r = radon_cancer_risk(8.0, smoker=False)
    >>> round(r.value, 5)       # 8 * 0.7e-3 = 0.0056
    0.0056

    Same concentration, smoker:

    >>> r = radon_cancer_risk(8.0, smoker=True)
    >>> round(r.value, 4)       # 8 * 7e-3 = 0.056
    0.056

    References
    ----------
    US EPA (2003). Assessment of Risks from Radon in Homes.
    EPA 402-R-03-003.

    BEIR VI Committee (1999). Health Effects of Exposure to Radon.
    National Research Council. National Academy Press.

    WHO (2009). WHO Handbook on Indoor Radon: A Public Health
    Perspective.

    Notes
    -----
    Quote: "The ground beneath you exhales."

    Radon is the #1 cause of lung cancer among non-smokers in the
    US (EPA estimate 21,000 deaths/year). Testing is cheap ($15-30
    for a kit); mitigation typically $800-2500 for sub-slab
    depressurization. Remarkable benefit-to-cost ratio for a
    toxin with no ambient-outdoor regulation.
    """
    u = unit.lower().strip().replace(" ", "").replace("^", "").replace("³", "3")
    u = u.replace("µ", "u")

    R = np.atleast_1d(np.asarray(radon_level, dtype=float))
    if np.any(R < 0):
        raise ValueError("radon_level must be non-negative.")
    if exposure_years <= 0:
        raise ValueError("exposure_years must be > 0.")

    if u in ("pci/l", "pcil"):
        pci = R
        bq = R * _BQ_M3_PER_PCIL
    elif u in ("bq/m3", "bqm3"):
        pci = R / _BQ_M3_PER_PCIL
        bq = R
    else:
        raise ValueError(f"unit must be 'pCi/L' or 'Bq/m3', got {unit!r}.")

    key = "smoker" if smoker else "never_smoker"
    per_pcil = _EPA_LIFETIME_RISK_PER_PCIL[key]
    # Scale by exposure years (EPA 2003 uses 70-year default)
    risk = pci * per_pcil * (exposure_years / 70.0)

    val = float(risk.mean()) if risk.size > 1 else float(risk.item())

    epa_action = pci > 4.0
    who_reference = pci > 2.7

    return DescriptiveResult(
        name="radon_cancer_risk",
        value=val,
        extra={
            "lifetime_cancer_risk": (risk.tolist() if risk.size > 1
                                      else float(risk.item())),
            "radon_pCi_per_L": pci.tolist() if pci.size > 1 else float(pci.item()),
            "radon_Bq_per_m3": bq.tolist() if bq.size > 1 else float(bq.item()),
            "smoker": smoker,
            "exposure_years": exposure_years,
            "risk_factor_per_pCiL": per_pcil,
            "above_epa_action_level": (epa_action.tolist() if epa_action.size > 1
                                        else bool(epa_action.item())),
            "above_who_reference_level": (who_reference.tolist() if who_reference.size > 1
                                           else bool(who_reference.item())),
            "source": "US EPA 402-R-03-003 / BEIR VI 1999 / WHO 2009",
        },
    )


radon = radon_cancer_risk


def cheatsheet() -> str:
    return "radon(pCi_L, smoker=False) -> lifetime lung-cancer excess risk."
