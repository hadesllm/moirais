# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""AdS/CFT dictionary: conformal dimension <-> bulk mass."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ads_cft_dictionary(
    delta: float | None = None,
    m_squared: float | None = None,
    d: int = 4,
    L: float = 1.0,
) -> DescriptiveResult:
    r"""Apply the AdS/CFT mass-dimension relation.

    For a scalar field in :math:`AdS_{d+1}`:

    .. math::

        m^2 L^2 = \\Delta(\\Delta - d)

    Given :math:`\\Delta`, compute :math:`m^2`; or given :math:`m^2`,
    compute :math:`\\Delta`.

    :param delta: Conformal dimension of boundary operator.
    :param m_squared: Bulk mass squared (in units of 1/L^2).
    :param d: Boundary spacetime dimension.
    :param L: AdS radius. Must be > 0.
    :return: DescriptiveResult with delta, m_squared.
    """
    if L <= 0:
        raise ValueError(f"AdS radius must be > 0, got {L}.")
    if delta is not None:
        m2 = delta * (delta - d) / L**2
        return DescriptiveResult(
            name="ads_cft_dictionary",
            value=float(delta),
            extra={"delta": float(delta), "m_squared": float(m2), "d": d, "L": L},
        )
    elif m_squared is not None:
        discriminant = d**2 / 4.0 + m_squared * L**2
        if discriminant < 0:
            raise ValueError("BF bound violated: m^2 L^2 < -d^2/4.")
        delta_plus = d / 2.0 + np.sqrt(discriminant)
        delta_minus = d / 2.0 - np.sqrt(discriminant)
        return DescriptiveResult(
            name="ads_cft_dictionary",
            value=float(delta_plus),
            extra={
                "delta_plus": float(delta_plus),
                "delta_minus": float(delta_minus),
                "m_squared": float(m_squared),
                "d": d,
                "L": L,
                "bf_bound": float(-(d**2) / (4 * L**2)),
            },
        )
    else:
        raise ValueError("Provide either delta or m_squared.")


def cheatsheet() -> str:
    return "ads_cft_dictionary(delta, m_squared, d) -> AdS/CFT mass-dimension"


adscf = ads_cft_dictionary
