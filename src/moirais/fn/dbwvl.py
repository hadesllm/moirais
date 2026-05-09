# moirais.fn — function file (hadesllm/moirais)
"""Generate Daubechies wavelet filter coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Much to learn you still have."


_DB_FILTERS = {
    1: np.array([1.0, 1.0]) / np.sqrt(2),
    2: np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2),
    3: np.array([0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175]) / np.sqrt(2),
    4: np.array([0.32580343, 1.01094572, 0.8922014, -0.03957503, -0.26450717, 0.0436163, 0.0465036, -0.01498699])
    / np.sqrt(2),
    5: np.array(
        [
            0.22641898,
            0.85394354,
            1.02432694,
            0.19576696,
            -0.34265671,
            -0.04560113,
            0.10970265,
            -0.00882680,
            -0.01779187,
            0.00471743,
        ]
    )
    / np.sqrt(2),
    6: np.array(
        [
            0.15774243,
            0.69950381,
            1.06226376,
            0.44583132,
            -0.31998660,
            -0.18351806,
            0.13788809,
            0.03892321,
            -0.04466375,
            0.00078781,
            0.00675606,
            -0.00152353,
        ]
    )
    / np.sqrt(2),
    7: np.array(
        [
            0.11009943,
            0.56079128,
            1.03114849,
            0.66437248,
            -0.20351382,
            -0.31683501,
            0.10084600,
            0.11400345,
            -0.05378245,
            -0.02343994,
            0.01774979,
            0.00006858,
            -0.00254790,
            0.00050028,
        ]
    )
    / np.sqrt(2),
    8: np.array(
        [
            0.07695562,
            0.44246725,
            0.95548615,
            0.82781653,
            -0.02238574,
            -0.40165863,
            0.00016961,
            0.18207636,
            -0.02456390,
            -0.06235021,
            0.01977216,
            0.01236884,
            -0.00688771,
            -0.00055429,
            0.00095522,
            -0.00016640,
        ]
    )
    / np.sqrt(2),
}


def daubechies_wavelet(order: int = 4) -> DescriptiveResult:
    """Generate Daubechies wavelet filter coefficients.

    Parameters
    ----------
    order : int
        Order N (db1 through db8 supported).

    Returns
    -------
    DescriptiveResult
    """
    if order not in _DB_FILTERS:
        raise ValueError(f"Supported orders: 1-8, got {order}")
    lo = _DB_FILTERS[order].copy()
    hi = np.array([(-1) ** k * lo[len(lo) - 1 - k] for k in range(len(lo))])
    return DescriptiveResult(
        name="daubechies_wavelet",
        value=float(order),
        extra={"lo_d": lo, "hi_d": hi, "order": order, "length": len(lo)},
    )


dbwvl = daubechies_wavelet


def cheatsheet() -> str:
    return "daubechies_wavelet({}) -> Generate Daubechies wavelet filter coefficients."
