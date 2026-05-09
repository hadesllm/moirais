# moirais.fn — function file (hadesllm/moirais)
"""Procrustes rotation for configuration alignment."""

from __future__ import annotations

from ._containers import DescriptiveResult


def procrustes_rotation(X, X_target) -> DescriptiveResult:
    """Orthogonal Procrustes rotation to align X to a target.

    :param X: Configuration to rotate (n x dims).
    :param X_target: Target configuration (n x dims).
    :return: DescriptiveResult with rotated matrix and MSE in ``extra``.

    .. epigraph:: "Power comes in response to a need." -- Goku, Dragon Ball Z
    """
    from moirais._spatial_voting import procrustes_rotation as _fn

    result = _fn(X, X_target)
    return DescriptiveResult(
        name="procrustes_rotation",
        value=result["mse"],
        extra=result,
    )


procr = procrustes_rotation


def cheatsheet() -> str:
    return "procrustes_rotation({}) -> Procrustes rotation for configuration alignment."
