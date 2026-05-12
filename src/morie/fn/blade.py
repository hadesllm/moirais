# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Edge detection (Canny variant). 'Some motherfuckers are always trying to ice-skate uphill.' -- Blade"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def edge_detect(
    image: np.ndarray,
    *,
    sigma: float = 1.0,
    low_threshold: float = 0.1,
    high_threshold: float = 0.3,
) -> DescriptiveResult:
    """Detect edges in a 2D image using a simplified Canny algorithm.

    1. Gaussian smoothing
    2. Sobel gradient magnitude and direction
    3. Non-maximum suppression
    4. Double-threshold hysteresis

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (float, [0, 1] or [0, 255]).
    sigma : float
        Gaussian smoothing standard deviation.
    low_threshold, high_threshold : float
        Fraction of max gradient for hysteresis thresholds.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``edges`` (binary 2D array), ``gradient_magnitude``,
        ``n_edge_pixels``.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be 2D")
    if img.max() > 1.0:
        img = img / 255.0

    k_size = max(3, int(6 * sigma) | 1)
    ax = np.arange(-k_size // 2 + 1, k_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    from scipy.signal import convolve2d

    smoothed = convolve2d(img, kernel, mode="same", boundary="symm")

    Gx = convolve2d(smoothed, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode="same", boundary="symm")
    Gy = convolve2d(smoothed, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode="same", boundary="symm")
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx)

    h, w = G.shape
    nms = np.zeros_like(G)
    angle = (np.rad2deg(theta) % 180).astype(int)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            if a < 22 or a >= 157:
                n1, n2 = G[i, j - 1], G[i, j + 1]
            elif a < 67:
                n1, n2 = G[i - 1, j + 1], G[i + 1, j - 1]
            elif a < 112:
                n1, n2 = G[i - 1, j], G[i + 1, j]
            else:
                n1, n2 = G[i - 1, j - 1], G[i + 1, j + 1]
            nms[i, j] = G[i, j] if G[i, j] >= n1 and G[i, j] >= n2 else 0

    gmax = nms.max()
    if gmax > 0:
        low = low_threshold * gmax
        high = high_threshold * gmax
    else:
        low = high = 0

    strong = nms >= high
    weak = (nms >= low) & ~strong
    edges = strong.copy()

    changed = True
    while changed:
        changed = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak[i, j] and np.any(edges[i - 1 : i + 2, j - 1 : j + 2]):
                    edges[i, j] = True
                    weak[i, j] = False
                    changed = True

    return DescriptiveResult(
        name="edge_detect",
        value={
            "edges": edges.astype(int),
            "gradient_magnitude": G,
            "n_edge_pixels": int(edges.sum()),
        },
        extra={"sigma": sigma, "shape": img.shape},
    )


blade = edge_detect


def cheatsheet() -> str:
    return "edge_detect({}) -> Edge detection (Canny variant). 'Some motherfuckers are alwa"
