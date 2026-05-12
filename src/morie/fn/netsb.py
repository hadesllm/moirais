# morie.fn -- function file (hadesllm/morie)
"""Bootstrap stability of network edge weights (CS coefficient)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def network_stability(
    data: pd.DataFrame | np.ndarray,
    *,
    n_boot: int = 100,
    seed: int = 42,
) -> dict:
    """Case-dropping bootstrap stability of edge weights.

    The Correlation Stability (CS) coefficient is the maximum proportion
    of cases that can be dropped such that the correlation between the
    original and bootstrapped centrality remains >= 0.7 in 95% of
    bootstraps (Epskamp et al., 2018).

    Parameters
    ----------
    data : DataFrame or ndarray
        Item-level data (n x p).
    n_boot : int
        Number of bootstrap iterations (default 100).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``cs_strength``, ``cs_ei``, ``edge_ci_lower`` (matrix),
        ``edge_ci_upper`` (matrix), ``n_boot``.

    References
    ----------
    Epskamp, S., Borsboom, D., & Fried, E. I. (2018). Estimating
    psychological networks and their accuracy. *Behavior Research
    Methods*, 50(1), 195--212.
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
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

    orig_net = _pcor(X)
    orig_strength = np.sum(np.abs(orig_net), axis=1)
    orig_ei = np.sum(orig_net, axis=1)

    # Edge weight CIs
    edge_boots = np.zeros((n_boot, p, p))
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        edge_boots[b] = _pcor(X[idx])

    edge_ci_lo = np.percentile(edge_boots, 2.5, axis=0)
    edge_ci_hi = np.percentile(edge_boots, 97.5, axis=0)

    # CS coefficient: case-dropping
    drop_fracs = np.arange(0.1, 0.95, 0.05)
    cs_strength = 0.0
    cs_ei = 0.0

    for frac in drop_fracs:
        n_sub = max(p + 1, int(n * (1 - frac)))
        cor_str = []
        cor_ei = []
        for _ in range(min(n_boot, 50)):
            idx = rng.choice(n, n_sub, replace=False)
            sub_net = _pcor(X[idx])
            sub_str = np.sum(np.abs(sub_net), axis=1)
            sub_ei = np.sum(sub_net, axis=1)
            cor_str.append(np.corrcoef(orig_strength, sub_str)[0, 1])
            cor_ei.append(np.corrcoef(orig_ei, sub_ei)[0, 1])

        if np.percentile(cor_str, 5) >= 0.7:
            cs_strength = frac
        if np.percentile(cor_ei, 5) >= 0.7:
            cs_ei = frac

    return {
        "cs_strength": float(cs_strength),
        "cs_ei": float(cs_ei),
        "edge_ci_lower": edge_ci_lo,
        "edge_ci_upper": edge_ci_hi,
        "n_boot": n_boot,
    }


def cheatsheet() -> str:
    return "network_stability({}) -> Bootstrap stability of network edge weights (CS coefficient)"
