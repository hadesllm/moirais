# morie.fn — function file (hadesllm/morie)
"""Material stress-strain curve analysis. 'I will break you.' -- Colossus"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def stress_strain(
    strain: np.ndarray | list[float],
    stress: np.ndarray | list[float],
    *,
    yield_offset: float = 0.002,
) -> DescriptiveResult:
    """Analyse a stress-strain curve to extract material properties.

    Computes Young's modulus (slope of linear region), yield stress
    (0.2% offset method), ultimate tensile strength, toughness (area
    under curve), and resilience (area under elastic region).

    Parameters
    ----------
    strain : array-like
        Engineering strain values (dimensionless).
    stress : array-like
        Engineering stress values (same units, e.g. MPa).
    yield_offset : float
        Offset strain for yield point determination (default 0.002 = 0.2%).

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``youngs_modulus``, ``yield_stress``,
        ``uts`` (ultimate tensile strength), ``toughness``, ``resilience``.
    """
    eps = np.asarray(strain, dtype=float)
    sig = np.asarray(stress, dtype=float)
    if len(eps) != len(sig) or len(eps) < 3:
        raise ValueError("strain and stress must have equal length >= 3")

    order = np.argsort(eps)
    eps = eps[order]
    sig = sig[order]

    n_linear = max(3, len(eps) // 5)
    E, intercept = np.polyfit(eps[:n_linear], sig[:n_linear], 1)
    E = float(E)

    offset_line = E * (eps - yield_offset)
    diff = sig - offset_line
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        t = diff[idx] / (diff[idx] - diff[idx + 1]) if abs(diff[idx] - diff[idx + 1]) > 1e-30 else 0.5
        yield_stress = float(sig[idx] + t * (sig[idx + 1] - sig[idx]))
    else:
        yield_stress = float(sig[n_linear])

    uts = float(np.max(sig))
    toughness = float(np.trapezoid(sig, eps))

    yield_idx = np.searchsorted(sig, yield_stress)
    yield_idx = min(yield_idx, len(eps) - 1)
    resilience = float(np.trapezoid(sig[: yield_idx + 1], eps[: yield_idx + 1]))

    return DescriptiveResult(
        name="stress_strain",
        value={
            "youngs_modulus": E,
            "yield_stress": yield_stress,
            "uts": uts,
            "toughness": toughness,
            "resilience": resilience,
        },
        extra={"n_points": len(eps), "yield_offset": yield_offset},
    )


colsm = stress_strain


def cheatsheet() -> str:
    return "stress_strain({}) -> Material stress-strain curve analysis. 'I will break you.' -"
