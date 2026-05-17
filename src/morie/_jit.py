"""Acceleration layer for morie's numeric hot paths.

Since v0.9.1 these kernels are implemented in the compiled C++ core
(``morie._core``, built from ``libmorie/``). This module is a thin
dispatch shim: it coerces inputs to the kernel ABI and calls the C++
implementation when the core is available, falling back to an
equivalent pure-numpy implementation otherwise (e.g. a source checkout
without the built extension). The two paths are numerically equivalent.

The numba JIT path was retired in v0.9.1 -- numba is no longer a
dependency or an optional extra.

Callers import unchanged, e.g.::

    from morie._jit import normal_pdf

Kernels:
  - normal_pdf / normal_logpdf   -- the kernel inside dnorm/pnorm
  - mean_jit / std_jit / var_jit -- summary statistics
  - cor_pearson_jit              -- Pearson correlation
  - euclid_dist_jit              -- pairwise L2 distance
  - trimmed_ipw_weights_jit      -- clipped IPW weights
  - bootstrap_mean_jit           -- bootstrap replicate means (numpy;
        the C++ port is deferred pending a deliberate RNG decision)
"""
from __future__ import annotations

import math

import numpy as np

try:
    from . import _core as _c
    _CORE_AVAILABLE = True
except ImportError:  # pragma: no cover -- source checkout w/o built ext
    _CORE_AVAILABLE = False

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _vec(a) -> np.ndarray:
    """Coerce to a contiguous 1-D float64 array (the C++ kernel ABI)."""
    return np.ascontiguousarray(a, dtype=np.float64).ravel()


def normal_pdf(x, mean: float, sd: float) -> np.ndarray:
    """Normal PDF over an array -- the kernel inside dnorm."""
    x = _vec(x)
    if _CORE_AVAILABLE:
        return _c.normal_pdf(x, float(mean), float(sd))
    inv_sigma = 1.0 / sd
    z = (x - mean) * inv_sigma
    return inv_sigma * _INV_SQRT_2PI * np.exp(-0.5 * z * z)


def normal_logpdf(x, mean: float, sd: float) -> np.ndarray:
    """Normal log-density -- preferred for likelihoods (avoids underflow)."""
    x = _vec(x)
    if _CORE_AVAILABLE:
        return _c.normal_logpdf(x, float(mean), float(sd))
    inv_sigma = 1.0 / sd
    z = (x - mean) * inv_sigma
    return -math.log(sd) - _LOG_SQRT_2PI - 0.5 * z * z


def mean_jit(arr) -> float:
    """Arithmetic mean of a 1-D array."""
    arr = _vec(arr)
    if _CORE_AVAILABLE:
        return _c.mean_jit(arr)
    return float(np.mean(arr)) if arr.size else float("nan")


def var_jit(arr, ddof: int = 1) -> float:
    """Sample variance with optional ddof."""
    arr = _vec(arr)
    if _CORE_AVAILABLE:
        return _c.var_jit(arr, int(ddof))
    if arr.size - ddof <= 0:
        return float("nan")
    return float(np.var(arr, ddof=ddof))


def std_jit(arr, ddof: int = 1) -> float:
    """Sample standard deviation with optional ddof."""
    arr = _vec(arr)
    if _CORE_AVAILABLE:
        return _c.std_jit(arr, int(ddof))
    return math.sqrt(var_jit(arr, ddof))


def cor_pearson_jit(x, y) -> float:
    """Pearson correlation coefficient."""
    x, y = _vec(x), _vec(y)
    if _CORE_AVAILABLE:
        return _c.cor_pearson_jit(x, y)
    if x.size != y.size or x.size < 2:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(x, y)[0, 1]
    return float(r)


def euclid_dist_jit(a, b) -> float:
    """Euclidean (L2) distance between two equal-length vectors."""
    a, b = _vec(a), _vec(b)
    if _CORE_AVAILABLE:
        return _c.euclid_dist_jit(a, b)
    if a.size != b.size:
        return float("nan")
    return float(np.sqrt(np.sum((a - b) ** 2)))


def trimmed_ipw_weights_jit(treat, propensity, trim_lo: float = 0.01,
                            trim_hi: float = 0.99) -> np.ndarray:
    """IPW weights with propensity-score clipping at [trim_lo, trim_hi]."""
    treat, propensity = _vec(treat), _vec(propensity)
    if _CORE_AVAILABLE:
        return _c.trimmed_ipw_weights_jit(
            treat, propensity, float(trim_lo), float(trim_hi))
    e = np.clip(propensity, trim_lo, trim_hi)
    return np.where(treat == 1.0, 1.0 / e, 1.0 / (1.0 - e))


def bootstrap_mean_jit(arr, B: int, seed: int) -> np.ndarray:
    """Bootstrap-replicate means: ``B`` resamples of ``len(arr)`` drawn
    with replacement, returning each replicate's mean.

    Runs in the compiled C++ core (``std::mt19937_64``); a given seed
    is fully reproducible. Unlike the other kernels there is no
    pure-numpy fallback -- a different RNG would silently change the
    replicate values -- so the compiled core (``morie._core``) is
    required.
    """
    if not _CORE_AVAILABLE:
        raise RuntimeError(
            "bootstrap_mean_jit requires the compiled morie C++ core "
            "(morie._core); build the extension or install a wheel.")
    return _c.bootstrap_mean_jit(_vec(arr), int(B), int(seed))


def is_jit_available() -> bool:
    """True when the compiled C++ core (``morie._core``) is available."""
    return _CORE_AVAILABLE


__all__ = [
    "normal_pdf",
    "normal_logpdf",
    "mean_jit",
    "var_jit",
    "std_jit",
    "cor_pearson_jit",
    "euclid_dist_jit",
    "bootstrap_mean_jit",
    "trimmed_ipw_weights_jit",
    "is_jit_available",
]
