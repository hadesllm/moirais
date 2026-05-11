"""Wet Bulb Globe Temperature (ISO 7243 heat-stress index)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def wet_bulb_globe_temp(
    T_air_C: float | np.ndarray,
    T_wet_C: float | np.ndarray,
    T_globe_C: float | np.ndarray,
    *,
    outdoor: bool = True,
) -> DescriptiveResult:
    """Compute Wet Bulb Globe Temperature (WBGT) in °C.

    The gold-standard heat-stress index used by ISO 7243, ACGIH, NIOSH,
    and most occupational heat-safety programs. Accounts for
    temperature (T_air), humidity (via wet-bulb), and radiant heat
    (via globe temperature) simultaneously — strictly more informative
    than the NWS Heat Index, which only uses T_air + RH.

    Parameters
    ----------
    T_air_C : float or array-like
        Ambient (dry-bulb) air temperature, °C.
    T_wet_C : float or array-like
        Natural wet-bulb temperature, °C. Measure with a
        wetted-wick thermometer in natural (not forced) air flow.
    T_globe_C : float or array-like
        Black globe thermometer reading, °C. Standard 150 mm black
        copper globe.
    outdoor : bool, default True
        If True (sunlit/outdoor exposure), WBGT = 0.7·Twb + 0.2·Tg
        + 0.1·Ta. If False (indoor / no solar load), WBGT = 0.7·Twb
        + 0.3·Tg.

    Returns
    -------
    DescriptiveResult
        value = WBGT in °C (mean if arrays). extra has the full
        per-observation series plus a threshold classification
        per ACGIH light-work limits:
        - safe:      WBGT ≤ 28
        - caution:   28 < WBGT ≤ 30
        - danger:    30 < WBGT ≤ 32
        - extreme:   WBGT > 32

    References
    ----------
    ISO 7243:2017. "Ergonomics of the thermal environment —
    Assessment of heat stress using the WBGT index."

    Notes
    -----
    Quote: "I must complete my mission." — BB-8
    """
    Ta = np.atleast_1d(np.asarray(T_air_C, dtype=float))
    Tw = np.atleast_1d(np.asarray(T_wet_C, dtype=float))
    Tg = np.atleast_1d(np.asarray(T_globe_C, dtype=float))

    if not (Ta.shape == Tw.shape == Tg.shape):
        raise ValueError("T_air_C, T_wet_C, T_globe_C must match in shape.")

    if outdoor:
        W = 0.7 * Tw + 0.2 * Tg + 0.1 * Ta
    else:
        W = 0.7 * Tw + 0.3 * Tg

    def _classify(w: float) -> str:
        if w <= 28.0:
            return "safe"
        if w <= 30.0:
            return "caution"
        if w <= 32.0:
            return "danger"
        return "extreme"

    classes = [_classify(float(w)) for w in W.ravel()]
    val = float(W.mean()) if W.size > 1 else float(W.item())

    return DescriptiveResult(
        name="wbgt",
        value=val,
        extra={
            "wbgt_C": W.tolist() if W.size > 1 else float(W.item()),
            "classification": classes if W.size > 1 else classes[0],
            "outdoor": outdoor,
        },
    )


wbgt = wet_bulb_globe_temp


def cheatsheet() -> str:
    return "wbgt(Ta, Tw, Tg, outdoor=True) -> Wet-Bulb Globe Temperature (°C)."
