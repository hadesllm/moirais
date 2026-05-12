# morie.fn -- function file (hadesllm/morie)
"""Scree plot data for MDS dimensionality. 'Hollow Purple.' -- Gojo, Jujutsu Kaisen"""

from __future__ import annotations

from ._containers import DescriptiveResult


def scree_plot_data(D, max_dims=6):
    """Compute stress values across dimensionalities for a scree plot.

    Parameters
    ----------
    D : array-like
        Distance matrix (n x n).
    max_dims : int
        Maximum dimensionality to try.

    Returns
    -------
    DescriptiveResult
        value = dict with dims and stress_values.
    """
    import numpy as np

    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    max_dims = min(max_dims, n - 1)

    B = np.eye(n) - np.ones((n, n)) / n
    D2 = D**2
    G = -0.5 * B @ D2 @ B

    vals, vecs = np.linalg.eigh(G)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    dims = list(range(1, max_dims + 1))
    stresses = []
    triu = np.triu_indices(n, k=1)
    obs = D[triu]

    for p in dims:
        L_pos = np.maximum(vals[:p], 0.0)
        Z = vecs[:, :p] * np.sqrt(L_pos)
        D_hat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((Z[i] - Z[j]) ** 2))
                D_hat[i, j] = d
                D_hat[j, i] = d
        mod = D_hat[triu]
        num = np.sum((obs - mod) ** 2)
        denom = np.sum(obs**2)
        stresses.append(float(np.sqrt(num / denom)) if denom > 0 else 0.0)

    return DescriptiveResult(
        name="scree_plot_data", value={"dims": dims, "stress_values": stresses}, extra={"max_dims": max_dims}
    )


scree = scree_plot_data


def cheatsheet() -> str:
    return "scree_plot_data({}) -> Scree plot data for MDS dimensionality. 'Hollow Purple.' -- "
