# moirais.fn — function file (hadesllm/moirais)
"""Knowing yourself is the beginning of all wisdom. — Aristotle"""

from __future__ import annotations

from ._containers import DescriptiveResult


def line_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> DescriptiveResult:
    """
    Compute the intersection point of two line segments (p1-p2) and (p3-p4).

    Uses the parametric form:

    .. math::

        t = \\frac{(x_1 - x_3)(y_3 - y_4) - (y_1 - y_3)(x_3 - x_4)}
                  {(x_1 - x_2)(y_3 - y_4) - (y_1 - y_2)(x_3 - x_4)}

    Returns the intersection point if segments cross, or None.

    :param p1: First endpoint of segment 1.
    :param p2: Second endpoint of segment 1.
    :param p3: First endpoint of segment 2.
    :param p4: Second endpoint of segment 2.
    :return: DescriptiveResult with intersection point or None.
    :rtype: DescriptiveResult

    References
    ----------
    Goldman R. (2005). Intersection of Two Lines in Three-Space.
    *Graphics Gems I*, Academic Press.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return DescriptiveResult(
            name="line_intersect",
            value=None,
            extra={"intersects": False, "parallel": True},
        )
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return DescriptiveResult(
            name="line_intersect",
            value=(ix, iy),
            extra={"intersects": True, "t": t, "u": u, "point": (ix, iy)},
        )
    return DescriptiveResult(
        name="line_intersect",
        value=None,
        extra={"intersects": False, "t": t, "u": u},
    )


linit = line_intersect


def cheatsheet() -> str:
    return "line_intersect({}) -> Line segment intersection. 'This is where the fun begins.' -"
