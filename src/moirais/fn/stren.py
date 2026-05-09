"""String tension T = 1/(2*pi*alpha')."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def string_tension(
    alpha_prime: float = 1.0,
) -> DescriptiveResult:
    """Compute the fundamental string tension.

    .. math::

        T = \\frac{1}{2\\pi\\alpha'}

    The string tension sets the energy scale of the theory. The string
    length is :math:`l_s = \\sqrt{\\alpha'}`.

    :param alpha_prime: Regge slope (string length squared). Must be > 0.
    :return: DescriptiveResult with string tension and related scales.
    """
    if alpha_prime <= 0:
        raise ValueError(f"alpha_prime must be > 0, got {alpha_prime}.")
    tension = 1.0 / (2.0 * np.pi * alpha_prime)
    l_s = np.sqrt(alpha_prime)
    m_s = 1.0 / l_s
    return DescriptiveResult(
        name="string_tension",
        value=float(tension),
        extra={
            "tension": float(tension),
            "alpha_prime": alpha_prime,
            "string_length": float(l_s),
            "string_mass": float(m_s),
        },
    )


def cheatsheet() -> str:
    return "string_tension(alpha_prime) -> T = 1/(2*pi*alpha')"


stren = string_tension
