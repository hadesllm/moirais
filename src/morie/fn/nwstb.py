# morie.fn -- function file (hadesllm/morie)
"""Network stability via case-dropping bootstrap."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def network_stability(
    data: np.ndarray,
    *,
    n_boot: int = 500,
    prop_drop: float = 0.5,
    seed: int = 42,
) -> DescriptiveResult:
    """Network stability assessment via case-dropping bootstrap.

    Computes the correlation between the full-sample partial
    correlation network and networks from subsamples.

    Parameters
    ----------
    data : ndarray
        Item response matrix (n x p).
    n_boot : int
        Number of bootstrap samples (default 500).
    prop_drop : float
        Proportion of cases to drop (default 0.5).
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        value=dict with CS-coefficient and correlation distribution.

    References
    ----------
    Epskamp, S., Borsboom, D., & Fried, E. I. (2018). Estimating
    psychological networks and their accuracy. Behavior Research
    Methods, 50(1), 195-212.
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    n_keep = max(int(n * (1 - prop_drop)), p + 1)

    def partial_cor(D):
        R = np.corrcoef(D, rowvar=False)
        try:
            P = np.linalg.inv(R + np.eye(p) * 1e-6)
        except np.linalg.LinAlgError:
            return np.zeros((p, p))
        d = np.sqrt(np.diag(P))
        d[d < 1e-10] = 1.0
        pc = -P / np.outer(d, d)
        np.fill_diagonal(pc, 0.0)
        return pc

    full_net = partial_cor(X)
    full_vec = full_net[np.triu_indices(p, k=1)]

    rng = np.random.default_rng(seed)
    cors = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n_keep, replace=False)
        boot_net = partial_cor(X[idx])
        boot_vec = boot_net[np.triu_indices(p, k=1)]
        if np.std(full_vec) > 1e-10 and np.std(boot_vec) > 1e-10:
            cors[b] = np.corrcoef(full_vec, boot_vec)[0, 1]
        else:
            cors[b] = 1.0

    cs_coeff = float(np.mean(cors > 0.7))

    return DescriptiveResult(
        name="Network stability",
        value={
            "CS_coefficient": cs_coeff,
            "mean_cor": float(np.mean(cors)),
            "median_cor": float(np.median(cors)),
        },
        extra={
            "n_boot": n_boot,
            "prop_drop": prop_drop,
            "quantiles": {
                "q025": float(np.percentile(cors, 2.5)),
                "q975": float(np.percentile(cors, 97.5)),
            },
        },
    )


net_stability = network_stability


def cheatsheet() -> str:
    return "network_stability({}) -> Network stability via case-dropping bootstrap."
