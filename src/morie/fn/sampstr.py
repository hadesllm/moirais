# morie.fn — function file (hadesllm/morie)
"""Stratified random sample (proportional allocation)."""

from typing import Sequence, Union
import numpy as np
def sampstr(strata: dict, n_total: int, seed: int = 42) -> dict:
    """Stratified sample with proportional allocation.

    `strata` maps stratum name → list of population units. Returns
    same-keyed dict with sampled units; sample sizes proportional to
    stratum sizes.
    """
    rng = np.random.default_rng(seed)
    sizes = {k: len(v) for k, v in strata.items()}
    total = sum(sizes.values())
    if total == 0 or n_total <= 0:
        raise ValueError("invalid strata or n_total.")
    out = {}
    for k, units in strata.items():
        n_k = max(1, round(n_total * sizes[k] / total))
        n_k = min(n_k, sizes[k])
        idx = rng.choice(sizes[k], size=n_k, replace=False)
        out[k] = [units[i] for i in idx]
    return out
