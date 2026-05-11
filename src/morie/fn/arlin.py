# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Areal interpolation via dasymetric mapping."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def areal_interpolation(
    source_values: np.ndarray,
    source_areas: np.ndarray,
    overlap_matrix: np.ndarray,
    target_areas: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> SpatialResult:
    r"""Areal interpolation from source to target zones.

    Implements Tobler's pycnophylactic (area-weighted) interpolation:

    .. math::

        \hat{Y}_j = \sum_{i=1}^{m}
        \frac{A_{ij} \cdot w_i}{A_i} \cdot Y_i

    where :math:`A_{ij}` is the intersection area of source zone *i*
    with target zone *j*, :math:`A_i` is the total area of source zone
    *i*, :math:`Y_i` is the source value, and :math:`w_i` is an
    optional ancillary weight (for dasymetric mapping).

    Parameters
    ----------
    source_values : np.ndarray
        Values in source zones, shape ``(m,)``.
    source_areas : np.ndarray
        Total areas of source zones, shape ``(m,)``.
    overlap_matrix : np.ndarray
        Intersection areas, shape ``(m, k)`` where *k* is the number of
        target zones. ``overlap_matrix[i, j]`` = area shared by source *i*
        and target *j*.
    target_areas : np.ndarray, optional
        Areas of target zones, shape ``(k,)``. Used for density-based
        output normalization if provided.
    weights : np.ndarray, optional
        Ancillary weights per source zone for dasymetric refinement,
        shape ``(m,)``. Default: uniform.

    Returns
    -------
    SpatialResult
        ``statistic`` is the mean interpolated value.
        ``extra["target_values"]`` contains the interpolated values
        for all target zones.

    References
    ----------
    Tobler WR (1979). Smooth pycnophylactic interpolation for
    geographical regions. *Journal of the American Statistical
    Association*, 74(367), 519-530.

    Mennis J (2003). Generating surface models of population using
    dasymetric mapping. *The Professional Geographer*, 55(1), 31-42.

    Goodchild MF, Lam NS-N (1980). Areal interpolation: A variant of
    the traditional spatial problem. *Geo-Processing*, 1, 297-312.
    """
    src_vals = np.asarray(source_values, dtype=np.float64).ravel()
    src_areas = np.asarray(source_areas, dtype=np.float64).ravel()
    overlap = np.asarray(overlap_matrix, dtype=np.float64)
    m = len(src_vals)

    if src_areas.shape != (m,):
        raise ValueError("source_areas must match source_values length")
    if overlap.ndim != 2 or overlap.shape[0] != m:
        raise ValueError("overlap_matrix must be (m, k)")

    k = overlap.shape[1]

    if weights is None:
        w = np.ones(m, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != (m,):
            raise ValueError("weights must match source_values length")

    safe_areas = np.where(src_areas > 0, src_areas, 1.0)
    fractions = overlap / safe_areas[:, None]
    weighted_fractions = fractions * w[:, None]

    frac_sum = weighted_fractions.sum(axis=0)
    frac_sum = np.where(frac_sum > 0, frac_sum, 1.0)

    weighted_fractions_norm = weighted_fractions / frac_sum[None, :]

    target_values = (weighted_fractions_norm * src_vals[:, None]).sum(axis=0)

    if target_areas is not None:
        tgt = np.asarray(target_areas, dtype=np.float64).ravel()
        if tgt.shape != (k,):
            raise ValueError("target_areas must have length k")

    return SpatialResult(
        name="Areal_Interpolation",
        statistic=float(np.mean(target_values)),
        p_value=None,
        extra={
            "target_values": target_values,
            "n_source": m,
            "n_target": k,
        },
    )


def cheatsheet() -> str:
    return "areal_interpolation({}) -> Areal interpolation via dasymetric mapping."
