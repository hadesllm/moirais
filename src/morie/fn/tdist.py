"""Geodesic distance on a torus."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def torus_distance(
    p1: tuple[float, float] = (0.0, 0.0),
    p2: tuple[float, float] = (np.pi, np.pi),
    R: float = 3.0,
    r: float = 1.0,
) -> DescriptiveResult:
    """Approximate geodesic distance between two points on a torus.

    Points given as (u, v) in radians. Uses numerical integration of the
    first fundamental form along a path parameterised by t in [0,1]:
    u(t) = u1 + t*(u2-u1+k*2pi), v(t) = v1 + t*(v2-v1+m*2pi)
    for the shortest-wrapping choice of k, m.

    :param p1: (u1, v1) on the torus.
    :param p2: (u2, v2) on the torus.
    :param R: Major radius.
    :param r: Minor radius.
    :return: DescriptiveResult with geodesic distance estimate.
    """
    if R <= 0 or r <= 0:
        raise ValueError(f"Radii must be positive, got R={R}, r={r}.")
    if r > R:
        raise ValueError(f"r={r} must be <= R={R}.")
    u1, v1 = p1
    u2, v2 = p2

    def _wrap(a, b):
        d = (b - a) % (2 * np.pi)
        if d > np.pi:
            d -= 2 * np.pi
        return d

    du = _wrap(u1, u2)
    dv = _wrap(v1, v2)
    n_steps = 200
    ts = np.linspace(0, 1, n_steps)
    v_t = v1 + dv * ts
    u_dot = du
    v_dot = dv
    ds = np.sqrt((R + r * np.cos(v_t)) ** 2 * u_dot**2 + r**2 * v_dot**2)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    distance = float(_trapz(ds, ts))
    chord = np.sqrt(
        ((R + r * np.cos(v2)) * np.cos(u2) - (R + r * np.cos(v1)) * np.cos(u1)) ** 2
        + ((R + r * np.cos(v2)) * np.sin(u2) - (R + r * np.cos(v1)) * np.sin(u1)) ** 2
        + (r * np.sin(v2) - r * np.sin(v1)) ** 2
    )
    return DescriptiveResult(
        name="torus_distance",
        value=distance,
        extra={
            "geodesic": distance,
            "chord": float(chord),
            "p1": p1,
            "p2": p2,
            "R": R,
            "r": r,
        },
    )


def cheatsheet() -> str:
    return "torus_distance(p1, p2, R, r) -> geodesic distance on torus"
