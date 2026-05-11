# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Air quality index from pollutant concentrations."""

from ._containers import ESRes

_AQI_BREAKPOINTS = {
    "pm25": [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
    ],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150)],
    "o3": [(0, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150)],
}


def air_quality_index(
    pollutant_conc: dict[str, float],
) -> ESRes:
    """Compute AQI from pollutant concentrations.

    Uses US EPA breakpoint method for PM2.5, PM10, O3.

    Parameters
    ----------
    pollutant_conc : dict
        {pollutant: concentration}.

    Returns
    -------
    ESRes
    """
    sub_indices = {}
    for pollutant, conc in pollutant_conc.items():
        key = pollutant.lower().replace(".", "")
        if key in _AQI_BREAKPOINTS:
            for c_lo, c_hi, i_lo, i_hi in _AQI_BREAKPOINTS[key]:
                if c_lo <= conc <= c_hi:
                    aqi = (i_hi - i_lo) / (c_hi - c_lo) * (conc - c_lo) + i_lo
                    sub_indices[pollutant] = float(aqi)
                    break
            else:
                sub_indices[pollutant] = 301.0
        else:
            sub_indices[pollutant] = float(conc)

    overall = max(sub_indices.values()) if sub_indices else 0.0
    if overall <= 50:
        category = "good"
    elif overall <= 100:
        category = "moderate"
    elif overall <= 150:
        category = "unhealthy_sensitive"
    elif overall <= 200:
        category = "unhealthy"
    else:
        category = "very_unhealthy"

    return ESRes(
        measure="AQI",
        estimate=float(overall),
        extra={"sub_indices": sub_indices, "category": category},
    )


aqidx = air_quality_index


def cheatsheet() -> str:
    return "air_quality_index({}) -> Air quality index from pollutant concentrations."
