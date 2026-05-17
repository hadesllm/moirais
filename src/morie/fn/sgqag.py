"""Multi-scale quadrat aggregation analysis."""

from __future__ import annotations

from ._containers import DescriptiveResult


def quadrat_aggregation(counts_or_points, scales, window=None):
    """Compute dispersion index at multiple spatial scales.

    .. epigraph:: Knowledge is power. -- Francis Bacon

    Parameters
    ----------
    counts_or_points : array_like
        Either pre-computed counts per scale or point coordinates.
    scales : array_like
        Grid sizes (number of quadrats per side).
    window : tuple, optional
        ``(xmin, xmax, ymin, ymax)`` required if points are given.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    pts = np.asarray(counts_or_points, dtype=np.float64)
    scales = np.asarray(scales, dtype=int)

    vmrs = []
    for s in scales:
        if pts.ndim == 2 and pts.shape[1] == 2 and window is not None:
            xmin, xmax, ymin, ymax = window
            xedges = np.linspace(xmin, xmax, s + 1)
            yedges = np.linspace(ymin, ymax, s + 1)
            counts = np.zeros(s * s)
            for k in range(len(pts)):
                ix = min(int((pts[k, 0] - xmin) / (xmax - xmin) * s), s - 1)
                iy = min(int((pts[k, 1] - ymin) / (ymax - ymin) * s), s - 1)
                counts[iy * s + ix] += 1
        else:
            counts = pts.ravel()

        mean_c = counts.mean()
        var_c = counts.var(ddof=1) if len(counts) > 1 else 0.0
        vmr = var_c / mean_c if mean_c > 0 else 0.0
        vmrs.append(float(vmr))

    return DescriptiveResult(
        name="quadrat_aggregation",
        value=float(np.mean(vmrs)),
        extra={
            "scales": scales.tolist(),
            "VMR_values": vmrs,
            "mean_VMR": float(np.mean(vmrs)),
        },
    )


sgqag = quadrat_aggregation


def cheatsheet() -> str:
    return "quadrat_aggregation({}) -> Multi-scale quadrat aggregation analysis."
