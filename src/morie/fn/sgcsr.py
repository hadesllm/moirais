"""Complete spatial randomness (CSR) test."""

from __future__ import annotations

from ._containers import DescriptiveResult


def csr_test(points, window, n_sim=99):
    """Test for complete spatial randomness using nearest-neighbor distances.

    Compares the observed mean NN distance to simulated HPP envelopes.

    .. epigraph:: "You have my sword." -- Aragorn, Lord of the Rings

    Parameters
    ----------
    points : array_like
        Point coordinates, shape ``(n, 2)``.
    window : tuple
        ``(xmin, xmax, ymin, ymax)`` bounding box.
    n_sim : int
        Number of CSR simulations.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    xmin, xmax, ymin, ymax = window

    D = squareform(pdist(pts))
    np.fill_diagonal(D, np.inf)
    nn_dists = D.min(axis=1)
    obs_mean = float(nn_dists.mean())

    rng = np.random.default_rng(42)
    sim_means = np.empty(n_sim)
    for i in range(n_sim):
        sx = rng.uniform(xmin, xmax, n)
        sy = rng.uniform(ymin, ymax, n)
        spts = np.column_stack([sx, sy])
        sD = squareform(pdist(spts))
        np.fill_diagonal(sD, np.inf)
        sim_means[i] = sD.min(axis=1).mean()

    p_value = float(np.mean(sim_means <= obs_mean))
    area = (xmax - xmin) * (ymax - ymin)
    expected = 0.5 * np.sqrt(area / n)

    return DescriptiveResult(
        name="csr_test",
        value=obs_mean,
        extra={
            "observed_mean_nn": obs_mean,
            "expected_mean_nn": float(expected),
            "p_value": p_value,
            "R_index": obs_mean / expected if expected > 0 else 0.0,
            "n_points": n,
            "csr_rejected": p_value < 0.05,
        },
    )


sgcsr = csr_test


def cheatsheet() -> str:
    return "csr_test({}) -> Complete spatial randomness (CSR) test."
