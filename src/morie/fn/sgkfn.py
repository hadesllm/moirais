"""Ripley's K function estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def ripley_k_function(points, window, r_values=None, correction="ripley"):
    """Estimate Ripley's K function for a point pattern.

    .. epigraph:: In the midst of chaos, there is also opportunity. -- Sun Tzu

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.
    window : tuple
        ``(xmin, xmax, ymin, ymax)``.
    r_values : array_like, optional
        Distances at which to evaluate K.
    correction : str
        Edge correction method (``'ripley'`` or ``'none'``).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    xmin, xmax, ymin, ymax = window
    area = (xmax - xmin) * (ymax - ymin)
    lam = n / area

    if r_values is None:
        max_r = min(xmax - xmin, ymax - ymin) / 4
        r_values = np.linspace(0, max_r, 25)
    else:
        r_values = np.asarray(r_values, dtype=np.float64)

    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(pts))

    K = np.zeros(len(r_values))
    for ri, r in enumerate(r_values):
        total = 0.0
        for i in range(n):
            for j in range(n):
                if i != j and D[i, j] <= r:
                    if correction == "ripley":
                        dx = min(pts[i, 0] - xmin, xmax - pts[i, 0])
                        dy = min(pts[i, 1] - ymin, ymax - pts[i, 1])
                        e = min(dx, dy, D[i, j])
                        w = 1.0 if e >= D[i, j] else max(0.5, e / D[i, j])
                        total += 1.0 / w
                    else:
                        total += 1.0
        K[ri] = area * total / (n * (n - 1)) if n > 1 else 0.0

    K_csr = np.pi * r_values**2

    return DescriptiveResult(
        name="ripley_k_function",
        value=float(K[-1]) if len(K) > 0 else 0.0,
        extra={
            "r_values": r_values.tolist(),
            "K_values": K.tolist(),
            "K_csr": K_csr.tolist(),
            "intensity": lam,
        },
    )


sgkfn = ripley_k_function


def cheatsheet() -> str:
    return "ripley_k_function({}) -> Ripley's K function estimation."
