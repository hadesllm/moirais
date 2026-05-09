# moirais.fn — function file (hadesllm/moirais)
"""Network comparison test (global strength and structure)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def network_compare(
    data1: pd.DataFrame | np.ndarray,
    data2: pd.DataFrame | np.ndarray,
    *,
    n_perm: int = 500,
    seed: int = 42,
) -> dict:
    """Permutation-based network comparison test.

    Tests whether two networks differ in global strength and/or
    network structure (van Borkulo et al., 2017).

    Parameters
    ----------
    data1 : DataFrame or ndarray
        Group 1 item-level data (n1 x p).
    data2 : DataFrame or ndarray
        Group 2 item-level data (n2 x p).
    n_perm : int
        Number of permutations (default 500).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``global_strength_diff``, ``global_strength_p``,
        ``structure_diff``, ``structure_p``, ``n_perm``.

    References
    ----------
    van Borkulo, C. D., et al. (2017). Comparing network structures
    on three aspects. Unpublished manuscript / *Psychological Methods*.
    """
    X1 = np.asarray(data1, dtype=np.float64)
    X2 = np.asarray(data2, dtype=np.float64)
    n1, p = X1.shape
    n2 = X2.shape[0]
    rng = np.random.default_rng(seed)

    def _pcor(mat: np.ndarray) -> np.ndarray:
        R = np.corrcoef(mat, rowvar=False)
        try:
            P = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(R)
        D = np.sqrt(np.diag(P))
        D[D == 0] = 1.0
        pc = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                if i != j:
                    pc[i, j] = -P[i, j] / (D[i] * D[j])
        return pc

    def _global_strength(net: np.ndarray) -> float:
        return float(np.sum(np.abs(net[np.triu_indices(p, k=1)])))

    def _structure_diff(n1: np.ndarray, n2: np.ndarray) -> float:
        return float(np.max(np.abs(n1 - n2)))

    net1 = _pcor(X1)
    net2 = _pcor(X2)
    obs_gs = abs(_global_strength(net1) - _global_strength(net2))
    obs_str = _structure_diff(net1, net2)

    # Permutation test
    combined = np.vstack([X1, X2])
    gs_perm = np.zeros(n_perm)
    str_perm = np.zeros(n_perm)

    for b in range(n_perm):
        idx = rng.permutation(n1 + n2)
        pX1 = combined[idx[:n1]]
        pX2 = combined[idx[n1:]]
        pnet1 = _pcor(pX1)
        pnet2 = _pcor(pX2)
        gs_perm[b] = abs(_global_strength(pnet1) - _global_strength(pnet2))
        str_perm[b] = _structure_diff(pnet1, pnet2)

    gs_p = float(np.mean(gs_perm >= obs_gs))
    str_p = float(np.mean(str_perm >= obs_str))

    return {
        "global_strength_diff": float(obs_gs),
        "global_strength_p": gs_p,
        "structure_diff": float(obs_str),
        "structure_p": str_p,
        "n_perm": n_perm,
    }


def cheatsheet() -> str:
    return "network_compare({}) -> Network comparison test (global strength and structure)."
