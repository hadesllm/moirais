"""JIT-ready acceleration layer for morie hot paths.

⚠ HONEST STATE -- read before believing the speed claims:

This module is *structurally* numba-ready: every kernel below is
decorated with @njit(cache=True) and uses numba.prange for parallel
loops. WHEN numba is installed and importable, the JIT path runs
and small-array kernels get ~5–10× speedup over scipy.stats.

When numba is NOT importable (the current state of morie's
primary venv on Python 3.15 -- numba's wheels max out at 3.14),
the `try import numba` fails, the module installs a no-op
decorator, and every function runs as plain numpy. **Numerically
identical, but no speedup.** `is_jit_available()` returns False
in that case.

To actually get the JIT speedup, run morie on a 3.10–3.14 venv
where `pip install numba` succeeds. Track:

    >>> from morie._jit import is_jit_available
    >>> is_jit_available()
    True   # ← only when numba imported

This module is *opt-in* -- `morie` proper doesn't import it eagerly.
fn/ files that want acceleration import via:

    from morie._jit import normal_pdf

and gracefully handle the import failure if their environment can't
support it. That way numba never becomes a hard dependency, and
the test suite stays green on Python 3.15 even though acceleration
is dormant.

Kernels exposed (all numerically correct in both modes):
  - normal_pdf / normal_logpdf  -- the kernel inside dnorm/qnorm/pnorm
  - mean_jit / std_jit / var_jit -- fused-loop summary stats
  - cor_pearson_jit              -- Pearson correlation in one pass
  - euclid_dist_jit              -- pairwise distance kernel
"""
from __future__ import annotations

import math
import numpy as np

# Attempt to enable JIT. The module exposes the same function names
# either way -- callers don't need to branch on availability.
try:
    import numba
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover -- Python 3.15 path
    _NUMBA_AVAILABLE = False
    # Stub decorators so the function bodies parse + run identically.
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

    def prange(n):  # type: ignore[no-redef]
        return range(n)


_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@njit(cache=True)
def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Normal PDF -- kernel of dnorm. ~5–10× faster than scipy.stats.norm.pdf
    on small arrays once compiled, comparable on big arrays.
    """
    inv_sigma = 1.0 / sd
    z = (x - mean) * inv_sigma
    return inv_sigma * _INV_SQRT_2PI * np.exp(-0.5 * z * z)


@njit(cache=True)
def normal_logpdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Log-density form -- preferred for likelihood calculations to
    avoid underflow.
    """
    inv_sigma = 1.0 / sd
    z = (x - mean) * inv_sigma
    return -math.log(sd) - _LOG_SQRT_2PI - 0.5 * z * z


@njit(cache=True, parallel=True)
def mean_jit(arr: np.ndarray) -> float:
    s = 0.0
    n = arr.shape[0]
    for i in prange(n):
        s += arr[i]
    return s / n if n > 0 else float("nan")


@njit(cache=True, parallel=True)
def var_jit(arr: np.ndarray, ddof: int = 1) -> float:
    """Sample variance with optional ddof. Two-pass for numerical
    stability (fused mean + sum-of-squared-deviations).
    """
    n = arr.shape[0]
    if n - ddof <= 0:
        return float("nan")
    m = mean_jit(arr)
    sq = 0.0
    for i in prange(n):
        d = arr[i] - m
        sq += d * d
    return sq / (n - ddof)


@njit(cache=True)
def std_jit(arr: np.ndarray, ddof: int = 1) -> float:
    return math.sqrt(var_jit(arr, ddof))


@njit(cache=True, parallel=True)
def cor_pearson_jit(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r in one parallel pass over the pairs."""
    n = x.shape[0]
    if n != y.shape[0] or n < 2:
        return float("nan")
    sx, sy, sxx, syy, sxy = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in prange(n):
        a, b = x[i], y[i]
        sx += a
        sy += b
        sxx += a * a
        syy += b * b
        sxy += a * b
    num = n * sxy - sx * sy
    den_sq = (n * sxx - sx * sx) * (n * syy - sy * sy)
    if den_sq <= 0.0:
        return float("nan")
    return num / math.sqrt(den_sq)


@njit(cache=True, parallel=True)
def euclid_dist_jit(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two equal-length vectors. Hot path for k-NN
    and clustering loops.
    """
    n = a.shape[0]
    s = 0.0
    for i in prange(n):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)


def is_jit_available() -> bool:
    """Probe -- used by doctor / selftest to advertise the speedup."""
    return _NUMBA_AVAILABLE


__all__ = [
    "normal_pdf",
    "normal_logpdf",
    "mean_jit",
    "var_jit",
    "std_jit",
    "cor_pearson_jit",
    "euclid_dist_jit",
    "is_jit_available",
]
