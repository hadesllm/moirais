# moirais.fn — function file (hadesllm/moirais)
"""Humidex (Environment Canada apparent-temperature index)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def humidex(
    T_air_C: float | np.ndarray,
    T_dew_C: float | np.ndarray,
) -> DescriptiveResult:
    """Compute Humidex from air temperature and dewpoint, in °C.

    Environment Canada's apparent-temperature index, widely used across
    Canadian weather forecasts, occupational heat guidelines, and
    epidemiological studies. Unlike US Heat Index (which needs relative
    humidity), Humidex uses dewpoint directly.

    Formula (Masterton & Richardson 1979):

    .. math::

        H = T + \\frac{5}{9}(e - 10)

    where the partial water-vapor pressure is

    .. math::

        e = 6.11 \\exp\\!\\left( 5417.7530
        \\left( \\frac{1}{273.16} - \\frac{1}{T_d + 273.16} \\right) \\right)

    and :math:`T_d` is the dewpoint in °C.

    Parameters
    ----------
    T_air_C : float or array-like
        Air temperature, °C.
    T_dew_C : float or array-like
        Dewpoint temperature, °C. Must satisfy T_dew ≤ T_air.

    Returns
    -------
    DescriptiveResult
        value = Humidex (unitless but reported on a °C-like scale).
        extra has per-observation Humidex, vapor pressure e,
        and Environment Canada discomfort classification:
        - no discomfort:       < 30
        - some discomfort:      30–39
        - great discomfort:     40–45
        - dangerous:            46–53
        - heat stroke imminent: ≥ 54

    References
    ----------
    Masterton, J. M., & Richardson, F. A. (1979). Humidex: A method of
    quantifying human discomfort due to excessive heat and humidity.
    Environment Canada, Downsview, ON. CLI 1-79.

    Notes
    -----
    Quote: "The days are getting shorter." — Grogu
    """
    T = np.atleast_1d(np.asarray(T_air_C, dtype=float))
    Td = np.atleast_1d(np.asarray(T_dew_C, dtype=float))

    if T.shape != Td.shape:
        raise ValueError("T_air_C and T_dew_C must match in shape.")
    if np.any(Td > T + 1e-9):
        raise ValueError("T_dew_C must be ≤ T_air_C (physical constraint).")

    # Vapor pressure in hPa via Clausius-Clapeyron approximation
    e = 6.11 * np.exp(5417.7530 * (1.0 / 273.16 - 1.0 / (Td + 273.16)))
    H = T + (5.0 / 9.0) * (e - 10.0)

    def _classify(h: float) -> str:
        if h < 30.0:
            return "no discomfort"
        if h < 40.0:
            return "some discomfort"
        if h < 46.0:
            return "great discomfort"
        if h < 54.0:
            return "dangerous"
        return "heat stroke imminent"

    classes = [_classify(float(h)) for h in H.ravel()]
    val = float(H.mean()) if H.size > 1 else float(H.item())

    return DescriptiveResult(
        name="humidex",
        value=val,
        extra={
            "humidex": H.tolist() if H.size > 1 else float(H.item()),
            "vapor_pressure_hPa": e.tolist() if e.size > 1 else float(e.item()),
            "classification": classes if H.size > 1 else classes[0],
        },
    )


def cheatsheet() -> str:
    return "humidex(T_air_C, T_dew_C) -> Environment Canada humidex."
