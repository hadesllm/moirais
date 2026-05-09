"""Genus computation for surfaces."""

from __future__ import annotations

from ._containers import DescriptiveResult


def torus_genus(
    handles: int | None = None,
    vertices: int | None = None,
    edges: int | None = None,
    faces: int | None = None,
) -> DescriptiveResult:
    """Compute genus of a closed orientable surface.

    If *handles* given directly, genus = handles.
    If V, E, F given (triangulation), compute via Euler formula:
    :math:`g = 1 - (V - E + F) / 2`

    :param handles: Number of handles (direct genus).
    :param vertices: Vertex count of triangulation.
    :param edges: Edge count.
    :param faces: Face count.
    :return: DescriptiveResult with genus and Euler characteristic.
    """
    if handles is not None:
        if handles < 0:
            raise ValueError(f"handles must be >= 0, got {handles}.")
        g = handles
        chi = 2 - 2 * g
    elif vertices is not None and edges is not None and faces is not None:
        chi = vertices - edges + faces
        g = (2 - chi) // 2
        if (2 - chi) % 2 != 0 or g < 0:
            raise ValueError(f"V-E+F={chi} does not yield valid orientable genus.")
    else:
        raise ValueError("Provide either handles or (vertices, edges, faces).")
    return DescriptiveResult(
        name="torus_genus",
        value=float(g),
        extra={"genus": g, "euler_characteristic": chi},
    )


def cheatsheet() -> str:
    return "torus_genus(handles | V,E,F) -> genus of surface"
