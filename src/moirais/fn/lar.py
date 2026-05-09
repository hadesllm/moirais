# moirais.fn — function file (hadesllm/moirais)
"""Log Area Ratios from LPC coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


def log_area_ratio_fn(lpc_coeffs: np.ndarray) -> DescriptiveResult:
    """Compute Log Area Ratios (LAR) from LPC reflection coefficients.

    .. math::

        g_i = \\ln\\left(\\frac{1 - k_i}{1 + k_i}\\right)

    where :math:`k_i` are the reflection coefficients derived from LPC.

    :param lpc_coeffs: LPC coefficients [a1, ..., ap].
    :return: DescriptiveResult with log area ratio values.
    """
    from moirais._armodel import reflection_coefficients

    lpc_coeffs = np.asarray(lpc_coeffs, dtype=float).ravel()
    k = reflection_coefficients(lpc_coeffs)
    k_clipped = np.clip(k, -0.9999, 0.9999)
    lar_vals = np.log((1 - k_clipped) / (1 + k_clipped))
    return DescriptiveResult(
        name="log_area_ratio",
        value=None,
        extra={"lar": lar_vals, "reflection_coeffs": k, "order": len(lpc_coeffs)},
    )


lar = log_area_ratio_fn


def cheatsheet() -> str:
    return "log_area_ratio_fn({}) -> Log Area Ratios from LPC coefficients."
