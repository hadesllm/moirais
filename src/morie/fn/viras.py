"""Virasoro algebra commutation relations."""

from __future__ import annotations

from ._containers import DescriptiveResult


def virasoro_algebra(
    c: float = 26.0,
    m: int = 2,
    n: int = -2,
) -> DescriptiveResult:
    """Compute the Virasoro algebra commutation relation [L_m, L_n].

    .. math::

        [L_m, L_n] = (m - n) L_{m+n} + \\frac{c}{12} m(m^2 - 1) \\delta_{m+n,0}

    :param c: Central charge (26 for bosonic string, 15 for superstring).
    :param m: Mode index m.
    :param n: Mode index n.
    :return: DescriptiveResult with structure constant and anomaly term.
    """
    structure = m - n
    anomaly = (c / 12.0) * m * (m**2 - 1) if m + n == 0 else 0.0
    return DescriptiveResult(
        name="virasoro_algebra",
        value=float(structure),
        extra={
            "structure_constant": structure,
            "anomaly": float(anomaly),
            "central_charge": c,
            "m": m,
            "n": n,
            "m_plus_n": m + n,
        },
    )


def cheatsheet() -> str:
    return "virasoro_algebra(c, m, n) -> [L_m, L_n] Virasoro commutation"


viras = virasoro_algebra
