"""Veneziano amplitude B(-alpha(s), -alpha(t))."""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as _gamma

from ._containers import DescriptiveResult


def veneziano_amplitude(
    s: float = 1.0,
    t: float = -0.5,
    alpha0: float = 1.0,
    alpha_prime: float = 0.5,
) -> DescriptiveResult:
    """Compute the Veneziano amplitude A(s,t) = B(-alpha(s), -alpha(t)).

    The Veneziano amplitude is the Euler beta function of linear Regge
    trajectories:

    .. math::

        A(s,t) = B(-\\alpha(s), -\\alpha(t))
               = \\frac{\\Gamma(-\\alpha(s))\\,\\Gamma(-\\alpha(t))}
                       {\\Gamma(-\\alpha(s)-\\alpha(t))}

    where :math:`\\alpha(x) = \\alpha_0 + \\alpha' x`.

    :param s: Mandelstam variable *s*.
    :param t: Mandelstam variable *t*.
    :param alpha0: Regge intercept.
    :param alpha_prime: Regge slope.
    :return: DescriptiveResult with amplitude magnitude.
    """
    alpha_s = alpha0 + alpha_prime * s
    alpha_t = alpha0 + alpha_prime * t
    num = _gamma(-alpha_s) * _gamma(-alpha_t)
    den = _gamma(-alpha_s - alpha_t)
    amplitude = num / den
    return DescriptiveResult(
        name="veneziano_amplitude",
        value=float(np.abs(amplitude)),
        extra={
            "amplitude_complex": complex(amplitude),
            "alpha_s": alpha_s,
            "alpha_t": alpha_t,
            "s": s,
            "t": t,
        },
    )


def cheatsheet() -> str:
    return "veneziano_amplitude(s, t) -> Veneziano amplitude B(-alpha(s), -alpha(t))"


vnamp = veneziano_amplitude
