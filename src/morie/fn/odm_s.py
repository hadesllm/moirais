# morie.fn — function file (hadesllm/morie)
"""Standardize demographics to reference population."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def otis_demo_standardize(
    observed: np.ndarray,
    standard: np.ndarray,
    *,
    rates: np.ndarray | None = None,
) -> DescriptiveResult:
    """Direct age-standardization to a reference population.

    Parameters
    ----------
    observed : ndarray
        Observed counts per stratum.
    standard : ndarray
        Standard population per stratum.
    rates : ndarray, optional
        Observed rates per stratum. If None, computed from observed/standard.

    Returns
    -------
    DescriptiveResult
    """
    observed = np.asarray(observed, dtype=float)
    standard = np.asarray(standard, dtype=float)
    if rates is None:
        rates = observed / np.maximum(standard, 1e-10)
    else:
        rates = np.asarray(rates, dtype=float)
    std_weights = standard / max(np.sum(standard), 1e-10)
    standardized_rate = float(np.sum(rates * std_weights))
    crude_rate = float(np.sum(observed) / max(np.sum(standard), 1e-10))
    return DescriptiveResult(
        name="otis_demo_standardize",
        value=standardized_rate,
        extra={"standardized_rate": standardized_rate, "crude_rate": crude_rate, "stratum_rates": rates.tolist()},
    )


odm_s = otis_demo_standardize


def cheatsheet() -> str:
    return "otis_demo_standardize({}) -> Standardize demographics to reference population."
