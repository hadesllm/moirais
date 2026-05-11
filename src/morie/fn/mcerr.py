# morie.fn — function file (hadesllm/morie)
"""MCMC standard error."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ._richresult import RichResult


def mcmc_se(
    samples: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Monte Carlo standard error using batch means.

    :param samples: MCMC samples (1-D array).
    :return: Dictionary with mcse, mean, sd, n, n_batches.

    References
    ----------
    Flegal, J. M. & Jones, G. L. (2010). *Annals of Statistics*, 38(2).
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    mean_x = float(np.mean(x))
    sd_x = float(np.std(x, ddof=1)) if n > 1 else 0.0

    n_batches = max(int(np.sqrt(n)), 2)
    batch_size = n // n_batches
    if batch_size < 1:
        return RichResult(payload={"mcse": sd_x / np.sqrt(n) if n > 0 else 0.0, "mean": mean_x, "sd": sd_x, "n": n, "n_batches": 1})

    batch_means = []
    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size
        batch_means.append(float(np.mean(x[start:end])))

    batch_means = np.array(batch_means)
    mcse = float(np.std(batch_means, ddof=1) / np.sqrt(n_batches))

    return {
        "mcse": mcse,
        "mean": mean_x,
        "sd": sd_x,
        "n": n,
        "n_batches": n_batches,
    }


mcerr = mcmc_se


def cheatsheet() -> str:
    return "mcmc_se({}) -> MCMC standard error."
