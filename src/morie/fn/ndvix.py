# morie.fn — function file (hadesllm/morie)
"""Green-space (NDVI) exposure health benefit estimator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

# Pooled per-0.1-NDVI all-cause mortality and mental-health RRs from
# Rojas-Rueda et al. 2019 Lancet Planetary Health meta-analysis of
# 9 cohort studies (n = 8,324,652 person-years):
#
#   All-cause mortality:   RR 0.96 (95% CI 0.94-0.97) per 0.1 NDVI
#   Cardiovascular:        RR 0.95 (95% CI 0.90-1.00) per 0.1 NDVI
#
# Mental-health benefits, Twohig-Bennett & Jones 2018 (BMJ Open)
# meta-analysis of 143 studies:
#
#   Depression (SMD):      -0.15 per 0.1 NDVI (pooled effect size)
#   Self-rated MH (OR):    0.91 per 0.1 NDVI increase
#
# Systolic BP reduction:  -2.6 mmHg per 0.1 NDVI (Twohig-Bennett 2018)
_NDVI_RR_PER_0_1: dict[str, tuple[float, tuple[float, float]]] = {
    "all_cause":        (0.96, (0.94, 0.97)),
    "cardiovascular":   (0.95, (0.90, 1.00)),
    "depression":       (0.85, (0.76, 0.95)),   # OR from Twohig-Bennett
    "self_rated_mh":    (0.91, (0.88, 0.94)),
}


def ndvi_exposure_rr(
    ndvi: float | np.ndarray,
    *,
    reference_ndvi: float = 0.2,
    outcome: str = "all_cause",
) -> DescriptiveResult:
    r"""Estimate health benefit of green-space exposure via NDVI.

    NDVI (Normalized Difference Vegetation Index) summarizes local
    vegetation density from satellite imagery (MODIS, Landsat) on a
    scale from -1 (water) to +1 (dense forest). Typical urban
    residential: 0.1-0.3; parks: 0.4-0.7; suburban: 0.3-0.5.

    Log-linear RR per 0.1-unit NDVI increase. This is the
    dose-response of a meta-analysis; individual-study effects vary
    with the NDVI buffer radius (most use 300 m or 500 m around
    residence).

    .. math::

        RR(NDVI) = \\exp\\!\\left( \\ln(RR_{0.1}) \\cdot
        \\frac{NDVI - NDVI_{\\mathrm{ref}}}{0.1} \\right)

    Parameters
    ----------
    ndvi : float or array-like
        Residential NDVI (typically 300-500 m buffer mean).
    reference_ndvi : float, default 0.2
        Reference NDVI for the rate ratio (residential-urban
        baseline ≈ 0.2).
    outcome : {"all_cause", "cardiovascular", "depression",
               "self_rated_mh"}
        Health outcome.

    Returns
    -------
    DescriptiveResult
        value = RR at the mean NDVI (RR < 1 = protective).
        extra has per-observation RR, 95% CI, source citation, and
        the per-0.1 RR used.

    Examples
    --------
    A resident moving from low-green (NDVI 0.15) to park-adjacent
    (NDVI 0.45) — estimated all-cause mortality benefit:

    >>> r_low  = ndvi_exposure_rr(0.15, reference_ndvi=0.15)
    >>> r_high = ndvi_exposure_rr(0.45, reference_ndvi=0.15)
    >>> round(r_high.value, 4)    # 0.96^3 ≈ 0.885
    0.8847

    References
    ----------
    Rojas-Rueda, D., Nieuwenhuijsen, M. J., Gascon, M., Perez-Leon, D.,
    & Mudu, P. (2019). Green spaces and mortality: a systematic
    review and meta-analysis of cohort studies. Lancet Planetary
    Health, 3(11), e469-e477.

    Twohig-Bennett, C., & Jones, A. (2018). The health benefits of
    the great outdoors: a systematic review and meta-analysis of
    greenspace exposure and health outcomes. Environmental Research,
    166, 628-637.

    Notes
    -----
    Quote: "The best medicine is the one that comes with photosynthesis."

    Links to MORIE's mental-health / substance-use research focus:
    green-space exposure is one of the few environmental interventions
    with consistent MH benefits across studies, and spatial disparity
    in urban green cover often tracks demographic disparity.
    """
    key = outcome.lower().strip().replace(" ", "_").replace("-", "_")
    if key not in _NDVI_RR_PER_0_1:
        raise KeyError(
            f"Unknown outcome {outcome!r}. Available: "
            f"{sorted(_NDVI_RR_PER_0_1)}"
        )
    rr_per, (lo_per, hi_per) = _NDVI_RR_PER_0_1[key]

    N = np.atleast_1d(np.asarray(ndvi, dtype=float))
    if np.any(N < -1.01) or np.any(N > 1.01):
        raise ValueError(
            "NDVI values must be in [-1, 1] (slight numerical slack allowed)."
        )
    if not (-1.0 <= reference_ndvi <= 1.0):
        raise ValueError("reference_ndvi must be in [-1, 1].")

    delta = (N - reference_ndvi) / 0.1
    rr = np.exp(np.log(rr_per) * delta)
    rr_lo = np.exp(np.log(lo_per) * delta)
    rr_hi = np.exp(np.log(hi_per) * delta)

    val = float(rr.mean()) if rr.size > 1 else float(rr.item())

    return DescriptiveResult(
        name="ndvi_exposure_rr",
        value=val,
        extra={
            "rr": rr.tolist() if rr.size > 1 else float(rr.item()),
            "rr_95lo": rr_lo.tolist() if rr_lo.size > 1 else float(rr_lo.item()),
            "rr_95hi": rr_hi.tolist() if rr_hi.size > 1 else float(rr_hi.item()),
            "rr_per_0_1_ndvi": rr_per,
            "outcome": key,
            "reference_ndvi": reference_ndvi,
            "source": "Rojas-Rueda 2019 Lancet Planet Health / "
                       "Twohig-Bennett 2018 Environ Res",
        },
    )


ndvix = ndvi_exposure_rr


def cheatsheet() -> str:
    return "ndvix(ndvi, outcome='all_cause') -> green-space exposure RR."
