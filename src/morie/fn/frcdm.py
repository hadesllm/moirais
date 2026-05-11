# morie.fn — function file (hadesllm/morie)
"""Fractal dimension (box-counting)."""

import numpy as np

from ._containers import ESRes


def fractal_dimension(x, max_boxes: int = 20, **kwargs) -> ESRes:
    """
    Estimate fractal dimension via box-counting on a 1-D time series.

    Embeds the time series as (t, x(t)) in 2-D and counts the minimum
    number of boxes of size epsilon needed to cover the curve.

    :param x: 1-D array-like time series.
    :param max_boxes: Maximum number of box sizes to test.
    :return: ESRes with estimated fractal dimension.

    References
    ----------
    Mandelbrot BB (1982). The Fractal Geometry of Nature. Freeman.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < 4:
        raise ValueError("Need at least 4 observations.")

    t = np.arange(n, dtype=np.float64)
    t_range = t[-1] - t[0]
    x_range = np.max(x) - np.min(x)
    if x_range == 0:
        return ESRes(measure="fractal_dimension", estimate=1.0, n=n, extra={})

    t_norm = (t - t[0]) / t_range
    x_norm = (x - np.min(x)) / x_range

    epsilons = []
    counts = []
    for k in range(2, max_boxes + 1):
        eps = 1.0 / k
        grid_t = np.floor(t_norm / eps).astype(int)
        grid_x = np.floor(x_norm / eps).astype(int)
        grid_x = np.clip(grid_x, 0, k - 1)
        boxes = set(zip(grid_t, grid_x))
        epsilons.append(eps)
        counts.append(len(boxes))

    log_eps = np.log(1.0 / np.array(epsilons))
    log_cnt = np.log(np.array(counts, dtype=np.float64))
    A = np.column_stack([log_eps, np.ones(len(log_eps))])
    coeffs = np.linalg.lstsq(A, log_cnt, rcond=None)[0]
    fd = float(coeffs[0])

    return ESRes(
        measure="fractal_dimension",
        estimate=fd,
        n=n,
        extra={"method": "box_counting", "n_scales": len(epsilons)},
    )


frcdm = fractal_dimension


def cheatsheet() -> str:
    return "fractal_dimension(x) -> Box-counting fractal dimension."
