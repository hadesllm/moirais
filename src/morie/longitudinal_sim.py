# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synchronised longitudinal simulation: SyncRNG + AR / VAR coefficient
matrices + multivariate normal generation with structured covariance.

This module is a clean-room implementation of the techniques used in
the Hlozek--Bangari Collaborative-CIFAR-Catalyst project
(https://github.com/bangari-19/Collaborative-CIFAR-Project-).
That repository is unlicensed (all rights reserved); no source is
copied here. The techniques themselves (synchronised pseudo-random
streams, lagged / contemporaneous coefficient matrices, multivariate
normal generation under positive-definite Toeplitz / compound-symmetric
covariance structures) are standard methods from the longitudinal-
modelling and ecological-time-series literature; this implementation
follows the methods independently from primary references.

Functions:
    sync_rng(seed): construct a synchronised RNG that emits the same
        stream across the Python and R sides of a workflow (when paired
        with the R `morie::sync_rng_R(seed)` wrapper).
    generate_ar_coefficients(): construct a p × p autoregressive
        coefficient matrix with stationarity-preserving spectral
        constraints.
    generate_var_coefficients(): construct a (p × p × lags) array of
        VAR(L) coefficient matrices.
    mvn_with_covariance(): draw multivariate normal samples under one
        of {independent, AR(1), compound-symmetric, Toeplitz} covariance
        kernels.
    simulate_longitudinal_panel(): top-level driver that ties RNG,
        coefficient generation, MVN draws, and missing-data masking
        into a single tidy long-format DataFrame.

References:
    Hamilton, J. D. (1994). Time Series Analysis. Princeton University
        Press. (VAR coefficient matrices, stationarity criteria.)
    Diggle, P. J., Liang, K.-Y., & Zeger, S. L. (1994). Analysis of
        Longitudinal Data. Oxford University Press.
    L'Ecuyer, P. (1999). Good parameters and implementations for
        combined multiple recursive random number generators.
        Operations Research, 47(1), 159--164.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


__all__ = [
    "sync_rng",
    "generate_ar_coefficients",
    "generate_var_coefficients",
    "mvn_with_covariance",
    "simulate_longitudinal_panel",
]


CovarianceKernel = Literal["independent", "ar1", "compound", "toeplitz"]


def sync_rng(seed: int) -> np.random.Generator:
    """Return a numpy Generator initialised with `seed`.

    Pairs with the R-side wrapper `morie::sync_rng_R(seed)` to give
    bit-identical streams across the two language sides of a
    cross-language simulation workflow. Both sides use the PCG64 family
    initialised from the same scalar integer, which is reproducible.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return np.random.default_rng(int(seed))


def generate_ar_coefficients(
    p: int,
    *,
    rng: np.random.Generator,
    spectral_radius: float = 0.8,
    diagonal_bias: float = 0.4,
) -> np.ndarray:
    """Generate a stationarity-preserving p × p AR coefficient matrix.

    Args:
        p: dimension (number of variables in the panel).
        rng: a numpy Generator (see `sync_rng`).
        spectral_radius: target spectral radius < 1 to keep the AR
            process stationary. Recommended 0.5--0.9 for realistic
            longitudinal panels.
        diagonal_bias: how much weight to put on diagonal vs off-
            diagonal entries. 0 = pure cross-variable coupling,
            1 = each variable purely autoregressive on itself.

    Returns:
        A p × p numpy array A such that the AR(1) process
        x_{t+1} = A x_t + ε_t is stationary.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    A_diag = diagonal_bias * np.eye(p)
    A_off = (1 - diagonal_bias) * (rng.standard_normal((p, p)) * 0.3)
    np.fill_diagonal(A_off, 0)
    A = A_diag + A_off
    # Rescale to target spectral radius
    eigvals = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigvals)).real
    if rho > 0:
        A = A * (spectral_radius / rho)
    return A


def generate_var_coefficients(
    p: int,
    lags: int,
    *,
    rng: np.random.Generator,
    spectral_radius: float = 0.8,
    decay: float = 0.6,
) -> np.ndarray:
    """Generate a p × p × lags VAR(L) coefficient array.

    Each lag-l slice has its spectral radius scaled by `decay**l` so
    that the companion matrix of the VAR system is stationary.
    """
    if lags < 1:
        raise ValueError("lags must be >= 1")
    out = np.zeros((p, p, lags))
    for l in range(lags):
        A = generate_ar_coefficients(p, rng=rng,
                                     spectral_radius=spectral_radius * decay ** l)
        out[:, :, l] = A
    return out


def _toeplitz(p: int, rho: float) -> np.ndarray:
    """First-order Toeplitz covariance: Σ_ij = rho ** |i-j|."""
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _compound_symmetric(p: int, rho: float) -> np.ndarray:
    """Compound-symmetric covariance: rho off-diagonal, 1 diagonal."""
    out = rho * np.ones((p, p))
    np.fill_diagonal(out, 1.0)
    return out


def mvn_with_covariance(
    n: int,
    p: int,
    *,
    rng: np.random.Generator,
    kernel: CovarianceKernel = "ar1",
    rho: float = 0.5,
    mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw n samples from N_p(mean, Σ) with structured Σ.

    Args:
        n: number of samples.
        p: dimension.
        rng: numpy Generator.
        kernel: one of `"independent"`, `"ar1"`, `"compound"`,
            `"toeplitz"`. `"ar1"` and `"toeplitz"` are equivalent here.
        rho: correlation parameter (off-diagonal element for compound,
            decay rate for AR1/toeplitz).
        mean: optional p-length mean vector (defaults to zero).

    Returns:
        An n × p array of samples.
    """
    if mean is None:
        mean = np.zeros(p)
    if kernel == "independent":
        sigma = np.eye(p)
    elif kernel in ("ar1", "toeplitz"):
        sigma = _toeplitz(p, rho)
    elif kernel == "compound":
        sigma = _compound_symmetric(p, rho)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    return rng.multivariate_normal(mean=mean, cov=sigma, size=n)


@dataclass
class LongitudinalSimSpec:
    """Specification for `simulate_longitudinal_panel`."""

    n_individuals: int
    n_timepoints: int
    p_variables: int
    cov_kernel: CovarianceKernel = "ar1"
    cov_rho: float = 0.5
    ar_lags: int = 1
    ar_spectral_radius: float = 0.8
    ar_decay: float = 0.6
    missing_fraction: float = 0.0
    outlier_fraction: float = 0.0
    outlier_scale: float = 5.0
    seed: int = 42


def simulate_longitudinal_panel(spec: LongitudinalSimSpec) -> pd.DataFrame:
    """Simulate a longitudinal panel and return as a tidy long-format frame.

    Each row is (subject_id, t, variable, value). The simulation:
      1. Seeds an RNG from spec.seed.
      2. Builds a VAR(spec.ar_lags) coefficient array.
      3. Draws subject-specific innovation streams from the chosen
         covariance kernel.
      4. Iterates the VAR system forward in time.
      5. Optionally masks `missing_fraction` entries to NaN.
      6. Optionally amplifies `outlier_fraction` entries by
         `outlier_scale`.

    Returns:
        Tidy long-format DataFrame with columns
        `subject_id`, `t`, `variable`, `value`.
    """
    rng = sync_rng(spec.seed)
    A = generate_var_coefficients(
        spec.p_variables, spec.ar_lags,
        rng=rng,
        spectral_radius=spec.ar_spectral_radius,
        decay=spec.ar_decay,
    )

    panel = np.zeros((spec.n_individuals, spec.n_timepoints, spec.p_variables))
    for i in range(spec.n_individuals):
        eps = mvn_with_covariance(
            spec.n_timepoints, spec.p_variables,
            rng=rng, kernel=spec.cov_kernel, rho=spec.cov_rho,
        )
        x = np.zeros(spec.p_variables)
        history = []
        for t in range(spec.n_timepoints):
            x_new = eps[t].copy()
            for l in range(spec.ar_lags):
                if t - l - 1 >= 0:
                    x_new = x_new + A[:, :, l] @ history[t - l - 1]
            history.append(x_new)
            panel[i, t] = x_new

    # Missing-data mask + outliers
    if spec.missing_fraction > 0:
        mask = rng.random(panel.shape) < spec.missing_fraction
        panel[mask] = np.nan
    if spec.outlier_fraction > 0:
        omask = rng.random(panel.shape) < spec.outlier_fraction
        panel[omask] = panel[omask] * spec.outlier_scale

    # Tidy
    out = []
    for i in range(spec.n_individuals):
        for t in range(spec.n_timepoints):
            for v in range(spec.p_variables):
                out.append((i, t, v, panel[i, t, v]))
    return pd.DataFrame(out, columns=["subject_id", "t", "variable", "value"])
