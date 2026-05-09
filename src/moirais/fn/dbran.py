# moirais.fn — function file (hadesllm/moirais)
"""D-brane tension calculation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def d_brane_tension(
    p: int = 3,
    g_s: float = 0.1,
    alpha_prime: float = 1.0,
) -> DescriptiveResult:
    """Compute the tension of a D_p-brane.

    .. math::

        T_p = \\frac{1}{g_s (2\\pi)^p (\\alpha')^{(p+1)/2}}

    :param p: Spatial dimension of the brane (0 to 9).
    :param g_s: String coupling. Must be > 0.
    :param alpha_prime: String length squared. Must be > 0.
    :return: DescriptiveResult with brane tension.
    """
    if p < 0 or p > 9:
        raise ValueError(f"Brane dimension must be 0-9, got {p}.")
    if g_s <= 0 or alpha_prime <= 0:
        raise ValueError("g_s and alpha_prime must be > 0.")
    tension = 1.0 / (g_s * (2 * np.pi) ** p * alpha_prime ** ((p + 1) / 2.0))
    return DescriptiveResult(
        name="d_brane_tension",
        value=float(tension),
        extra={
            "p": p,
            "g_s": g_s,
            "alpha_prime": alpha_prime,
            "tension": float(tension),
            "worldvolume_dim": p + 1,
        },
    )


def cheatsheet() -> str:
    return "d_brane_tension(p, g_s, alpha_prime) -> D_p-brane tension"


dbran = d_brane_tension
