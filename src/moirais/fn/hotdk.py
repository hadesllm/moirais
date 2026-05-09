# moirais.fn — function file (hadesllm/moirais)
"""Hot-deck imputation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hot_deck_impute(
    data: np.ndarray,
    *,
    seed: int | None = None,
) -> DescriptiveResult:
    """Hot-deck imputation: fill missing with random observed donor.

    For each missing cell, randomly selects from observed values in that column.

    Parameters
    ----------
    data : (n, p) array with np.nan
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, p = data.shape
    rng = np.random.default_rng(seed)

    imp = data.copy()
    n_filled = 0
    for j in range(p):
        nan_mask = np.isnan(imp[:, j])
        n_miss_j = int(nan_mask.sum())
        if n_miss_j == 0:
            continue
        observed = imp[~nan_mask, j]
        if len(observed) == 0:
            continue
        donors = rng.choice(observed, size=n_miss_j, replace=True)
        imp[nan_mask, j] = donors
        n_filled += n_miss_j

    return DescriptiveResult(
        name="hot_deck",
        value=float(n_filled),
        extra={"n_filled": n_filled, "n": n, "p": p, "pct_missing": float(n_filled / (n * p) * 100)},
    )


hotdk = hot_deck_impute


def cheatsheet() -> str:
    return "hot_deck_impute({}) -> Hot-deck imputation."
