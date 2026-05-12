"""Urban Heat Island (UHI) index."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def urban_heat_island(
    T_urban_C: float | np.ndarray,
    T_rural_C: float | np.ndarray,
    *,
    timestamps: np.ndarray | None = None,
) -> DescriptiveResult:
    """Compute Urban Heat Island (UHI) index as T_urban − T_rural (°C).

    UHI is the excess temperature in urban areas relative to nearby
    rural reference stations. Typically reported as a nocturnal
    minimum-to-minimum difference (UHI peaks at night due to heat
    retention in built surfaces), but any matched-in-time comparison
    is valid.

    For cities, classical reference:
    - UHI < 2°C: weak
    - 2–5°C: moderate
    - > 5°C: strong (common in hot-dry summer nights, e.g., Phoenix)

    Parameters
    ----------
    T_urban_C : float or array-like
        Urban station temperature(s), °C.
    T_rural_C : float or array-like
        Rural reference temperature(s), °C. Must match T_urban in
        shape AND in time (pair observations sampled at the same
        instant).
    timestamps : array-like, optional
        Optional timestamps for per-observation UHI values in `extra`.

    Returns
    -------
    DescriptiveResult
        value = mean UHI (°C).
        extra contains: UHI series, min/max/p95, classification
        (weak/moderate/strong), and sample size.

    References
    ----------
    Oke, T. R. (1982). The energetic basis of the urban heat island.
    Quarterly Journal of the Royal Meteorological Society, 108(455),
    1–24.

    Notes
    -----
    Quote: "The city is a jungle." -- every urban ecologist, ever.
    """
    Tu = np.atleast_1d(np.asarray(T_urban_C, dtype=float))
    Tr = np.atleast_1d(np.asarray(T_rural_C, dtype=float))
    if Tu.shape != Tr.shape:
        raise ValueError("T_urban_C and T_rural_C must match in shape.")

    uhi = Tu - Tr
    mean_uhi = float(uhi.mean())

    if mean_uhi < 2.0:
        cls = "weak"
    elif mean_uhi < 5.0:
        cls = "moderate"
    else:
        cls = "strong"

    extra: dict[str, object] = {
        "uhi_C": uhi.tolist() if uhi.size > 1 else float(uhi.item()),
        "mean_uhi_C": mean_uhi,
        "min_C": float(uhi.min()),
        "max_C": float(uhi.max()),
        "p95_C": float(np.percentile(uhi, 95)) if uhi.size > 1 else mean_uhi,
        "classification": cls,
        "n_obs": int(uhi.size),
    }
    if timestamps is not None:
        ts = np.asarray(timestamps)
        if ts.shape != Tu.shape:
            raise ValueError("timestamps must match T_urban shape.")
        extra["timestamps"] = ts.tolist()

    return DescriptiveResult(name="uhi", value=mean_uhi, extra=extra)


uhi = urban_heat_island


def cheatsheet() -> str:
    return "uhi(T_urban, T_rural) -> Urban Heat Island excess (°C)."
