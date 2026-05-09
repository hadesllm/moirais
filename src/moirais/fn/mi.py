# moirais.fn — function file (hadesllm/moirais)
"""Mutual information estimation."""

import numpy as np

from ._containers import DescriptiveResult


def mutual_info(x, y, n_bins=20):
    """
    Estimate mutual information I(X;Y) via histogram binning.

    :param x: (n,) first variable.
    :param y: (n,) second variable.
    :param n_bins: Number of histogram bins per variable.
    :return: DescriptiveResult with MI in nats, normalized MI.

    References
    ----------
    Cover TM, Thomas JA (2006). Elements of Information Theory. 2nd ed.
    Wiley.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(x)

    hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    pxy = hist_xy / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi_val = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi_val += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    nmi = mi_val / np.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0

    return DescriptiveResult(
        name="mutual_info",
        value=float(mi_val),
        extra={
            "mi_nats": float(mi_val),
            "mi_bits": float(mi_val / np.log(2)),
            "normalized_mi": float(nmi),
            "h_x": float(hx),
            "h_y": float(hy),
            "n_bins": n_bins,
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "mutual_info({}) -> Mutual information estimation."
