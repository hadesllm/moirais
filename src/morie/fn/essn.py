# morie.fn -- function file (hadesllm/morie)
"""Effective sample size for MCMC chains."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from ._richresult import RichResult


def effective_sample_size(
    samples: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Effective sample size (ESS) via initial positive sequence estimator.

    :param samples: MCMC samples (1-D array).
    :return: Dictionary with ess, ess_ratio, n.

    References
    ----------
    Geyer, C. J. (1992). *Statistical Science*, 7(4), 473--483.
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    if n < 4:
        return RichResult(payload={"ess": float(n), "ess_ratio": 1.0, "n": n})

    mean_x = np.mean(x)
    centered = x - mean_x

    max_lag = n - 1
    acf = np.correlate(centered, centered, mode="full")[n - 1:]
    acf = acf / acf[0] if acf[0] > 0 else acf

    tau = 1.0
    for t in range(1, max_lag // 2):
        gamma_2t = acf[2 * t] if 2 * t < len(acf) else 0.0
        gamma_2t1 = acf[2 * t + 1] if 2 * t + 1 < len(acf) else 0.0
        pair_sum = gamma_2t + gamma_2t1
        if pair_sum < 0:
            break
        tau += 2 * pair_sum

    ess = n / tau if tau > 0 else float(n)
    ess = min(ess, float(n))

    return {
        "ess": float(ess),
        "ess_ratio": float(ess / n),
        "n": n,
    }


essn = effective_sample_size


def cheatsheet() -> str:
    return "effective_sample_size({}) -> Effective sample size for MCMC chains."
