# morie.fn — function file (hadesllm/morie)
"""Acute ozone short-term mortality relative risk (NMMAPS / APHENA)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

# Pooled daily mortality RRs per 10 ppb increase in ambient O3 (8-hour
# max or daily mean). Time-series studies using GAMs on multi-city
# data. RRs are small per unit but concern is high because O3 exposure
# is widespread and chronic.
#
#   NMMAPS (Bell 2004, JAMA 292:2372-2378):
#     All-cause: 1.0052 per 10 ppb 24-h mean  (95% CI 1.0027-1.0077)
#
#   APHENA (Katsouyanni 2009, Environ Health Perspect 117:1747-1754):
#     All-cause: 1.0031 per 10 µg/m³ 8-h max
#
#   Meta-analysis Gryparis 2004 + Levy 2005: ~0.4-0.5% per 10 ppb.
#
# Conversion: 1 ppb O3 ≈ 1.96 µg/m³ at 25°C / 1 atm.
_O3_ACUTE_RR_PER_10_PPB: dict[str, tuple[float, tuple[float, float]]] = {
    "all_cause":     (1.0052, (1.0027, 1.0077)),
    "cardiovascular":(1.0064, (1.0031, 1.0098)),
    "respiratory":   (1.0080, (1.0040, 1.0120)),
}

_PPB_TO_UGM3 = 1.96  # at 25°C, 1 atm


def o3_acute_rr(
    concentration: float | np.ndarray,
    *,
    reference: float = 0.0,
    outcome: str = "all_cause",
    unit: str = "ppb",
) -> DescriptiveResult:
    """Short-term ozone exposure relative risk for daily mortality.

    Log-linear scaling of published per-10-ppb RR estimates from
    NMMAPS (Bell 2004) and APHENA (Katsouyanni 2009) time-series
    multicity studies.

    .. math::

        RR(C) = \\exp\\!\\left( \\ln(RR_{10ppb}) \\cdot
        \\frac{C - C_{\\mathrm{ref}}}{10} \\right)

    Parameters
    ----------
    concentration : float or array-like
        Ambient O₃ concentration. 8-hour daily maximum or 24-hour
        mean; match the pooled study design.
    reference : float, default 0.0
        Counterfactual reference concentration (same unit).
    outcome : {"all_cause", "cardiovascular", "respiratory"}
        Outcome category.
    unit : {"ppb", "ug/m3"}, default "ppb"
        Measurement unit. 1 ppb ≈ 1.96 µg/m³ at STP; conversion is
        applied when unit="ug/m3".

    Returns
    -------
    DescriptiveResult
        value = RR at the mean concentration.
        extra includes per-observation RR, 95% CI series, the base
        RR used, and citation.

    Examples
    --------
    Toronto summer averaging 45 ppb daily, vs WHO AQG peak-season of
    30 ppb (60 µg/m³ / 1.96):

    >>> r = o3_acute_rr(45, reference=30, outcome="all_cause")
    >>> round(r.value, 4)  # (1.0052)^1.5 ≈ 1.0078
    1.0078

    References
    ----------
    Bell, M. L., McDermott, A., Zeger, S. L., et al. (2004). Ozone
    and short-term mortality in 95 US urban communities, 1987-2000.
    JAMA, 292(19), 2372-2378.

    Katsouyanni, K., Samet, J. M., Anderson, H. R., et al. (2009).
    Air pollution and health: a European and North American
    approach (APHENA). Research Report 142. Health Effects Institute.

    Notes
    -----
    Quote: "Respiration is our most fundamental transaction with the
    environment. We owe it careful measurement." — paraphrase of
    Bernardino Ramazzini.
    """
    key = outcome.lower().strip().replace(" ", "_").replace("-", "_")
    if key not in _O3_ACUTE_RR_PER_10_PPB:
        raise KeyError(
            f"Unknown outcome {outcome!r}. Available: "
            f"{sorted(_O3_ACUTE_RR_PER_10_PPB)}"
        )
    rr10, (lo10, hi10) = _O3_ACUTE_RR_PER_10_PPB[key]

    C = np.atleast_1d(np.asarray(concentration, dtype=float))
    ref = float(reference)

    u = unit.lower().strip().replace("µ", "u")
    if u not in ("ppb", "ug/m3", "ug/m^3"):
        raise ValueError(
            f"unit must be 'ppb' or 'ug/m3', got {unit!r}."
        )
    if u != "ppb":
        # Convert µg/m³ → ppb before applying the per-10-ppb scaling
        C = C / _PPB_TO_UGM3
        ref = ref / _PPB_TO_UGM3

    delta = (C - ref) / 10.0
    rr = np.exp(np.log(rr10) * delta)
    rr_lo = np.exp(np.log(lo10) * delta)
    rr_hi = np.exp(np.log(hi10) * delta)

    val = float(rr.mean()) if rr.size > 1 else float(rr.item())

    return DescriptiveResult(
        name="o3_acute_rr",
        value=val,
        extra={
            "rr": rr.tolist() if rr.size > 1 else float(rr.item()),
            "rr_95lo": rr_lo.tolist() if rr_lo.size > 1 else float(rr_lo.item()),
            "rr_95hi": rr_hi.tolist() if rr_hi.size > 1 else float(rr_hi.item()),
            "rr_per_10_ppb": rr10,
            "outcome": key,
            "reference_ppb": ref,
            "unit_input": unit,
            "source": "Bell 2004 (NMMAPS JAMA) / Katsouyanni 2009 (APHENA HEI)",
        },
    )


o3acut = o3_acute_rr


def cheatsheet() -> str:
    return "o3acut(C, outcome='all_cause', unit='ppb') -> acute O3 RR (NMMAPS)."
