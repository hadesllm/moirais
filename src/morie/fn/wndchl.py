"""Environment Canada Wind Chill index."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def wind_chill(
    T_air_C: float | np.ndarray,
    wind_kmh: float | np.ndarray,
) -> DescriptiveResult:
    r"""Compute Wind Chill index (°C) per the Environment Canada formula.

    The joint US/Canada 2001 wind-chill index, replacing the 1945
    Siple-Passel scale. Valid only for T ≤ 10°C and wind ≥ 4.8 km/h;
    outside this envelope the formula returns T_air unchanged.

    Formula:

    .. math::

        W = 13.12 + 0.6215\\,T - 11.37\\,v^{0.16}
            + 0.3965\\,T \\cdot v^{0.16}

    where :math:`T` is air temperature in °C and :math:`v` is wind
    speed at 10 m in km/h.

    Frostbite risk thresholds (EC):

    - W > −28: low risk
    - −28 ≥ W > −40: risk of frostbite in ≤ 30 min exposed skin
    - −40 ≥ W > −48: frostbite in ≤ 10 min
    - −48 ≥ W > −55: frostbite in ≤ 5 min
    - W ≤ −55: frostbite in < 2 min

    Parameters
    ----------
    T_air_C : float or array-like
        Air temperature, °C.
    wind_kmh : float or array-like
        Wind speed at 10 m, km/h.

    Returns
    -------
    DescriptiveResult
        value = mean wind chill (°C).
        extra has per-observation W, classification, and a flag for
        whether the formula envelope was respected.

    References
    ----------
    Environment Canada (2001). Canada's Wind Chill Index: public
    information. Joint US National Weather Service / Meteorological
    Service of Canada standard, adopted November 2001.

    Notes
    -----
    Quote: "Winter is coming." — generic Stark.
    """
    T = np.atleast_1d(np.asarray(T_air_C, dtype=float))
    V = np.atleast_1d(np.asarray(wind_kmh, dtype=float))
    if T.shape != V.shape:
        raise ValueError("T_air_C and wind_kmh must match in shape.")
    if np.any(V < 0):
        raise ValueError("wind_kmh must be non-negative.")

    in_envelope = (T <= 10.0) & (V >= 4.8)
    v016 = np.where(V > 0, np.power(np.maximum(V, 1e-9), 0.16), 0.0)
    W_raw = 13.12 + 0.6215 * T - 11.37 * v016 + 0.3965 * T * v016
    W = np.where(in_envelope, W_raw, T)

    def _classify(w: float) -> str:
        if w > -28.0:
            return "low risk"
        if w > -40.0:
            return "frostbite 30 min"
        if w > -48.0:
            return "frostbite 10 min"
        if w > -55.0:
            return "frostbite 5 min"
        return "frostbite < 2 min"

    classes = [_classify(float(w)) for w in W.ravel()]
    val = float(W.mean()) if W.size > 1 else float(W.item())

    return DescriptiveResult(
        name="wind_chill",
        value=val,
        extra={
            "wind_chill_C": W.tolist() if W.size > 1 else float(W.item()),
            "classification": classes if W.size > 1 else classes[0],
            "in_envelope": in_envelope.tolist()
                if in_envelope.size > 1 else bool(in_envelope.item()),
        },
    )


wndchl = wind_chill


def cheatsheet() -> str:
    return "wndchl(T_air_C, wind_kmh) -> Wind chill index (°C, EC 2001)."
