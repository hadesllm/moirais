"""Generate Symlet wavelet filter coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


_SYM_FILTERS = {
    2: np.array([-0.12940952, 0.22414387, 0.83651630, 0.48296291]),
    3: np.array([0.03522629, -0.08544127, -0.13501102, 0.45987750, 0.80689151, 0.33267055]),
    4: np.array([-0.07576571, -0.02963553, 0.49761867, 0.80373875, 0.29785780, -0.09921954, -0.01260397, 0.03222310]),
    5: np.array([0.02767, -0.02955, -0.04772, 0.11430, 0.09364, -0.58723, 0.71690, 0.27331, -0.03128, 0.00055]),
    6: np.array(
        [
            0.01540,
            0.00349,
            -0.11799,
            -0.04831,
            0.49107,
            0.78738,
            0.33790,
            -0.07263,
            -0.02106,
            0.04472,
            0.00178,
            -0.00780,
        ]
    ),
}


def symlet_wavelet(order: int = 4) -> DescriptiveResult:
    """Generate Symlet wavelet filter coefficients.

    Parameters
    ----------
    order : int
        Order (sym2-sym6 supported).

    Returns
    -------
    DescriptiveResult
    """
    if order not in _SYM_FILTERS:
        raise ValueError(f"Supported sym orders: {sorted(_SYM_FILTERS)}, got {order}")
    lo = _SYM_FILTERS[order].copy()
    hi = np.array([(-1) ** k * lo[len(lo) - 1 - k] for k in range(len(lo))])
    return DescriptiveResult(
        name="symlet_wavelet",
        value=float(order),
        extra={"lo_d": lo, "hi_d": hi, "order": order, "length": len(lo)},
    )


symwv = symlet_wavelet


def cheatsheet() -> str:
    return "symlet_wavelet({}) -> Generate Symlet wavelet filter coefficients."
