# morie.fn — function file (hadesllm/morie)
"""Classical MDS for spatial map of legislators (Armstrong Ch 7)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mds_spatial_map", "mdspl"]


def mds_spatial_map(x, k: int = 2):
    """Classical (Torgerson) multidimensional scaling.

    Builds Euclidean distances D from input x, then computes the
    double-centred Gram matrix B = -1/2 * J D^2 J  and returns its
    top-k eigen-decomposition. Stress-1 reports goodness of fit:
        Stress = sqrt( sum (d_ij - hat_d_ij)^2 / sum d_ij^2 )

    Parameters
    ----------
    x : (n, p) configuration matrix OR (n, n) symmetric distance matrix.
        A 1-D vector is treated as a single coordinate per unit.
    k : int
        Number of MDS dimensions to extract (default 2).

    Returns
    -------
    RichResult with keys: coords, eigenvalues, stress, k, n
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    if n < 2:
        return RichResult(payload={"coords": np.zeros((n, k)),
                                   "eigenvalues": np.zeros(k),
                                   "stress": np.nan, "k": k, "n": n,
                                   "method": "mds_classical"})
    # Decide: distance matrix vs configuration
    if x.shape[0] == x.shape[1] \
            and np.allclose(x, x.T, atol=1e-9) \
            and np.allclose(np.diag(x), 0):
        D = x
    else:
        diff = x[:, None, :] - x[None, :, :]
        D = np.sqrt(np.sum(diff ** 2, axis=-1))
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh((B + B.T) / 2)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    k_eff = min(k, n - 1)
    pos = np.maximum(eigvals[:k_eff], 0)
    coords = eigvecs[:, :k_eff] * np.sqrt(pos)
    # Stress-1
    Dh = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2,
                        axis=-1))
    denom = float(np.sum(D ** 2))
    stress = float(np.sqrt(np.sum((D - Dh) ** 2) / denom)) if denom > 0 \
        else np.nan
    return RichResult(
        title="Classical MDS (Torgerson)",
        summary_lines=[("Stress-1", stress), ("k", k_eff), ("n", n),
                       ("Top eigenvalues",
                        list(np.round(eigvals[:min(5, n)], 4)))],
        interpretation=(
            f"k={k_eff}-dim MDS map; Stress-1 = {stress:.4f} "
            f"(<0.20 = good fit, Kruskal 1964)."),
        payload={"coords": coords, "eigenvalues": eigvals,
                 "stress": stress, "k": int(k_eff), "n": int(n),
                 "method": "mds_classical"},
    )


mdspl = mds_spatial_map


def cheatsheet():
    return "mdspl: Classical MDS — Torgerson double-centring + eigen-decomp."


# CANONICAL TEST
# >>> r = mds_spatial_map(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]), k=2)
# >>> assert r["stress"] < 1e-6
