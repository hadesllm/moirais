# moirais.fn — function file (hadesllm/moirais)
"""Heat index calculation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def heat_index(
    temp_F: float | np.ndarray | list,
    humidity_pct: float | np.ndarray | list,
) -> DescriptiveResult:
    """
    Compute heat index from temperature (Fahrenheit) and relative humidity.

    Uses the Rothfusz regression equation adopted by the US NWS.

    Parameters
    ----------
    temp_F : float or array-like
        Temperature in degrees Fahrenheit.
    humidity_pct : float or array-like
        Relative humidity in percent (0-100).

    Returns
    -------
    DescriptiveResult
        value = heat index (F); extra has 'heat_index_C'.

    References
    ----------
    Rothfusz, L. P. (1990). The heat index equation. NWS Technical
    Attachment SR 90-23.
    """
    T = np.atleast_1d(np.asarray(temp_F, dtype=float))
    R = np.atleast_1d(np.asarray(humidity_pct, dtype=float))
    if T.shape != R.shape:
        raise ValueError("temp_F and humidity_pct must match in shape.")

    HI = np.where(
        T < 80,
        0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094),
        (
            -42.379
            + 2.04901523 * T
            + 10.14333127 * R
            - 0.22475541 * T * R
            - 6.83783e-3 * T**2
            - 5.481717e-2 * R**2
            + 1.22874e-3 * T**2 * R
            + 8.5282e-4 * T * R**2
            - 1.99e-6 * T**2 * R**2
        ),
    )

    HI_C = (HI - 32) * 5 / 9
    val = float(HI.mean()) if HI.size > 1 else float(HI.item())

    return DescriptiveResult(
        name="heat_index",
        value=val,
        extra={
            "heat_index_F": HI.tolist() if HI.size > 1 else float(HI.item()),
            "heat_index_C": HI_C.tolist() if HI_C.size > 1 else float(HI_C.item()),
        },
    )


htidx = heat_index


def cheatsheet() -> str:
    return "heat_index({}) -> Heat index calculation."
