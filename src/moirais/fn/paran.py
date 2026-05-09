# moirais.fn — function file (hadesllm/moirais)
"""Horn's parallel analysis for factor retention."""

from __future__ import annotations

import numpy as np
import pandas as pd


def paran(
    data: pd.DataFrame | np.ndarray,
    nsim: int = 100,
    seed: int = 42,
) -> int:
    """Horn's parallel analysis -- suggested number of factors.

    Compares observed eigenvalues to the 95th percentile of eigenvalues
    from random data of the same dimensions.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    nsim : int
        Number of random simulations (default 100).
    seed : int
        Random seed (default 42).

    Returns
    -------
    int
        Suggested number of factors (minimum 1).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape

    R = np.corrcoef(X, rowvar=False)
    obs = np.sort(np.linalg.eigvalsh(R))[::-1]

    sim = np.zeros((nsim, k))
    for i in range(nsim):
        Rs = np.corrcoef(rng.standard_normal((n, k)), rowvar=False)
        sim[i] = np.sort(np.linalg.eigvalsh(Rs))[::-1]

    thresh = np.percentile(sim, 95, axis=0)
    return max(int(np.sum(obs > thresh)), 1)


def cheatsheet() -> str:
    return "paran({}) -> Horn's parallel analysis for factor retention."
