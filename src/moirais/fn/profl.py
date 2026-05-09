# moirais.fn — function file (hadesllm/moirais)
"""Procrustes rotation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def procrustes(
    source: np.ndarray,
    target: np.ndarray,
) -> DescriptiveResult:
    """Orthogonal Procrustes rotation: rotate source to best match target.

    Minimises :math:`\\|\\mathbf{T} - \\mathbf{S} \\mathbf{R}\\|_F` over
    orthogonal matrices *R*.

    Parameters
    ----------
    source : ndarray (n, p)
        Matrix to rotate.
    target : ndarray (n, p)
        Target matrix.

    Returns
    -------
    DescriptiveResult
        ``value`` is the rotated source matrix.
        ``extra`` has ``rotation``, ``scale``, ``disparity``.
    """
    S = np.asarray(source, dtype=np.float64)
    T = np.asarray(target, dtype=np.float64)

    mu_s = S.mean(axis=0)
    mu_t = T.mean(axis=0)
    S_c = S - mu_s
    T_c = T - mu_t

    norm_s = np.linalg.norm(S_c)
    norm_t = np.linalg.norm(T_c)
    S_c /= max(norm_s, 1e-12)
    T_c /= max(norm_t, 1e-12)

    U, _, Vt = np.linalg.svd(T_c.T @ S_c)
    R = Vt.T @ U.T

    scale = norm_t / max(norm_s, 1e-12)
    rotated = (S - mu_s) @ R * scale + mu_t

    disparity = float(np.sum((T - rotated) ** 2))

    return DescriptiveResult(
        name="Procrustes",
        value=rotated,
        extra={
            "rotation": R,
            "scale": scale,
            "disparity": disparity,
        },
    )


profl = procrustes


def cheatsheet() -> str:
    return "procrustes({}) -> Orthogonal Procrustes rotation."
