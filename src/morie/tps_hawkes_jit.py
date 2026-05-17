"""Numba-JIT'd negative log-likelihood paths for ``tps_hawkes_advanced``.

Reason this exists: ``tps_hawkes_advanced._neg_loglik_general`` did a pure-
Python O(n²) inner loop over events; at n=21,160 the optimizer spent
hours per fit.  This module replaces that hot loop with @njit'd code
plus the well-known O(n) recursive update for the exponential kernel,
so a 21k-event 2019-contiguous Markovian fit lands in seconds and the
Weibull×sinusoidal non-Markovian fit lands in minutes.

Public:

    has_jit_path(kernel, baseline) -> bool
    neg_loglik_jit(theta, t, T, kernel, baseline) -> float

The caller (``_neg_loglik_general``) checks ``has_jit_path`` first and
falls through to the original NumPy implementation only for
parameter combinations no JIT path covers.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover -- env without numba
    HAS_NUMBA = False
    prange = range  # type: ignore[assignment]

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        # passthrough decorator so module still imports
        if args and callable(args[0]):
            return args[0]

        def _wrap(f):
            return f

        return _wrap


# Phase 2 (v0.9.1): the compiled C++ core (morie._core) provides the
# exponential-kernel / constant-baseline likelihood -- no numba needed
# for that path.
try:
    from . import _core as _core_ext

    HAS_CORE = True
except ImportError:  # pragma: no cover -- source checkout w/o built ext
    HAS_CORE = False


# ── Exponential kernel, constant baseline (Mohler 2011) ─────────────


@njit(cache=True, fastmath=False)
def _ll_exp_const(t: np.ndarray, T: float, a0: float,
                   eta: float, beta: float) -> float:
    # box constraints (mirror _neg_loglik_general)
    # beta ∈ [0.01, 100] avoids the well-known Hawkes spike-train
    # degeneracy where β -> ∞ pushes log-lik to +∞ on finite data.
    if not (1e-6 < eta < 0.999):
        return 1e12
    if not (0.05 < beta < 30.0):
        return 1e12
    if not (-20.0 < a0 < 20.0):
        return 1e12
    n = t.size
    nu = math.exp(a0)
    if not math.isfinite(nu) or nu <= 0.0:
        return 1e12
    log_sum = 0.0
    # A_i = β · Σ_{j<i} exp(-β(t_i - t_j))
    # recurrence: A_1 = 0; A_i = exp(-β·Δt_i) · (A_{i-1} + β)
    A = 0.0
    for i in range(n):
        if i == 0:
            lam_i = nu
        else:
            dt = t[i] - t[i - 1]
            A = math.exp(-beta * dt) * (A + beta)
            lam_i = nu + eta * A
        if not math.isfinite(lam_i) or lam_i <= 0.0:
            return 1e12
        log_sum += math.log(lam_i)
    # integral: ∫_0^T ν dt + η Σ (1 - exp(-β(T-t_i)))
    integral = nu * T
    for i in range(n):
        integral += eta * (1.0 - math.exp(-beta * (T - t[i])))
    if not math.isfinite(log_sum) or not math.isfinite(integral):
        return 1e12
    return -(log_sum - integral)


# ── Exponential kernel, sinusoidal baseline ─────────────────────────


@njit(cache=True, fastmath=False)
def _ll_exp_sin(t: np.ndarray, T: float,
                 a0: float, a1: float, a2: float, a3: float,
                 eta: float, beta: float,
                 grid: np.ndarray, grid_vals: np.ndarray) -> float:
    """Exponential kernel × sinusoidal baseline; ν integral via trapezoid grid."""
    if not (1e-6 < eta < 0.999):
        return 1e12
    if not (0.05 < beta < 30.0):
        return 1e12
    if not (-20.0 < a0 < 20.0):
        return 1e12
    n = t.size
    log_sum = 0.0
    A = 0.0
    two_pi_y = 2.0 * math.pi / 365.25
    T_safe = T if T > 1.0 else 1.0
    for i in range(n):
        nu_i = math.exp(a0 + a1 * (t[i] / T_safe)
                         + a2 * math.sin(two_pi_y * t[i])
                         + a3 * math.cos(two_pi_y * t[i]))
        if i == 0:
            lam_i = nu_i
        else:
            dt = t[i] - t[i - 1]
            A = math.exp(-beta * dt) * (A + beta)
            lam_i = nu_i + eta * A
        if lam_i <= 0.0:
            return 1e12
        log_sum += math.log(lam_i)
    # baseline integral via trapezoidal rule on supplied grid
    g_int = 0.0
    for k in range(grid.size - 1):
        g_int += 0.5 * (grid_vals[k] + grid_vals[k + 1]) * (grid[k + 1] - grid[k])
    integral = g_int
    for i in range(n):
        integral += eta * (1.0 - math.exp(-beta * (T - t[i])))
    return -(log_sum - integral)


# ── Weibull kernel, constant baseline ───────────────────────────────


@njit(cache=True, fastmath=False, parallel=True)
def _ll_weibull_const(t: np.ndarray, T: float, a0: float,
                       eta: float, alpha: float, lam: float) -> float:
    if not (1e-6 < eta < 0.999):
        return 1e12
    if not (0.05 < alpha < 20.0):
        return 1e12
    if not (1e-3 < lam < 1e3):
        return 1e12
    if not (-20.0 < a0 < 20.0):
        return 1e12
    n = t.size
    nu = math.exp(a0)
    # parallel across i; each i computes its own intensity from prior events
    lam_at = np.empty(n)
    for i in prange(n):
        s = 0.0
        for j in range(i):
            x = (t[i] - t[j]) / lam
            if x > 1e-12:
                z = x ** alpha
                if z < 700.0:
                    s += (alpha / lam) * (x ** (alpha - 1.0)) * math.exp(-z)
        lam_at[i] = nu + eta * s
    log_sum = 0.0
    for i in range(n):
        if lam_at[i] <= 0.0:
            return 1e12
        log_sum += math.log(lam_at[i])
    integral = nu * T
    for i in range(n):
        u = T - t[i]
        if u > 0.0:
            x = u / lam
            integral += eta * (1.0 - math.exp(-(x ** alpha)))
    return -(log_sum - integral)


# ── Weibull kernel, sinusoidal baseline ─────────────────────────────


@njit(cache=True, fastmath=False, parallel=False)
def _ll_weibull_sin(t: np.ndarray, T: float,
                     a0: float, a1: float, a2: float, a3: float,
                     eta: float, alpha: float, lam: float,
                     grid: np.ndarray, grid_vals: np.ndarray) -> float:
    if not (1e-6 < eta < 0.999):
        return 1e12
    if not (0.05 < alpha < 20.0):
        return 1e12
    if not (1e-3 < lam < 1e3):
        return 1e12
    if not (-20.0 < a0 < 20.0):
        return 1e12
    n = t.size
    two_pi_y = 2.0 * math.pi / 365.25
    T_safe = T if T > 1.0 else 1.0
    log_sum = 0.0
    for i in range(n):
        nu_i = math.exp(a0 + a1 * (t[i] / T_safe)
                         + a2 * math.sin(two_pi_y * t[i])
                         + a3 * math.cos(two_pi_y * t[i]))
        s = 0.0
        for j in range(i):
            x = (t[i] - t[j]) / lam
            if x > 1e-12:
                z = x ** alpha
                if z < 700.0:  # otherwise exp(-z) underflows; term ≈ 0
                    s += (alpha / lam) * (x ** (alpha - 1.0)) * math.exp(-z)
        lam_i = nu_i + eta * s
        if not math.isfinite(lam_i) or lam_i <= 0.0:
            return 1e12
        log_sum += math.log(lam_i)
    g_int = 0.0
    for k in range(grid.size - 1):
        g_int += 0.5 * (grid_vals[k] + grid_vals[k + 1]) * (grid[k + 1] - grid[k])
    integral = g_int
    for i in range(n):
        u = T - t[i]
        if u > 0.0:
            x = u / lam
            integral += eta * (1.0 - math.exp(-(x ** alpha)))
    return -(log_sum - integral)


# ── Gamma kernel, constant baseline ─────────────────────────────────


@njit(cache=True, fastmath=False)
def _ll_gamma_const(t: np.ndarray, T: float, a0: float,
                     eta: float, alpha: float, beta: float) -> float:
    if not (1e-6 < eta < 0.999):
        return 1e12
    if not (0.05 < alpha < 20.0):
        return 1e12
    if not (0.05 < beta < 30.0):
        return 1e12
    if not (-20.0 < a0 < 20.0):
        return 1e12
    n = t.size
    nu = math.exp(a0)
    log_sum = 0.0
    log_const = alpha * math.log(beta) - math.lgamma(alpha)
    for i in range(n):
        s = 0.0
        for j in range(i):
            u = t[i] - t[j]
            if u > 1e-300:
                log_d = log_const + (alpha - 1.0) * math.log(u) - beta * u
                s += math.exp(log_d)
        lam_i = nu + eta * s
        if lam_i <= 0.0:
            return 1e12
        log_sum += math.log(lam_i)
    # ∫ exact gamma CDF requires gammainc; approximate via 1 - exp(...) trapezoid is wrong.
    # Numba lacks scipy.gammainc; use the regularized lower incomplete via series:
    integral = nu * T
    for i in range(n):
        integral += eta * _gamma_cdf_regularized(alpha, beta * (T - t[i]))
    return -(log_sum - integral)


@njit(cache=True, fastmath=False)
def _gamma_cdf_regularized(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) via series (x ≤ a+1) or
    continued fraction (x > a+1).  Numerical Recipes 6.2."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        # series
        gln = math.lgamma(a)
        ap = a
        sum_ = 1.0 / a
        delta = sum_
        for _ in range(200):
            ap += 1.0
            delta *= x / ap
            sum_ += delta
            if abs(delta) < abs(sum_) * 1e-12:
                break
        return sum_ * math.exp(-x + a * math.log(x) - gln)
    # continued fraction (Lentz)
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, 201):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


# ── Lomax kernel, constant baseline ─────────────────────────────────


@njit(cache=True, fastmath=False)
def _ll_lomax_const(t: np.ndarray, T: float, a0: float,
                     eta: float, alpha: float, c: float) -> float:
    n = t.size
    nu = math.exp(a0)
    log_sum = 0.0
    log_const = math.log(alpha - 1.0) + (alpha - 1.0) * math.log(c)
    for i in range(n):
        s = 0.0
        for j in range(i):
            u = t[i] - t[j]
            log_d = log_const - alpha * math.log(u + c)
            s += math.exp(log_d)
        lam_i = nu + eta * s
        if lam_i <= 0.0:
            return 1e12
        log_sum += math.log(lam_i)
    integral = nu * T
    for i in range(n):
        u = T - t[i]
        if u > 0.0:
            integral += eta * (1.0 - (c / (u + c)) ** (alpha - 1.0))
    return -(log_sum - integral)


# ── Lomax kernel, sinusoidal baseline (power-law + circadian/weekly) ─


@njit(cache=True, fastmath=False, parallel=True)
def _ll_lomax_sin(t: np.ndarray, T: float,
                   a0: float, a1: float, a2: float, a3: float,
                   eta: float, alpha: float, c: float,
                   grid: np.ndarray, grid_vals: np.ndarray) -> float:
    """Omori-type power-law kernel h(u) = (α-1)·c^{α-1}·(u+c)^{-α} (norm. density)
    with sinusoidal baseline.  Recommended for fat-tailed criminological
    excitation where exponential/Weibull misspecify (KS p ≪ 0.05)."""
    if not (1e-6 < eta < 0.999):
        return 1e12
    if not (1.05 < alpha < 30.0):
        return 1e12  # α > 1 needed for finite mean
    if not (1e-4 < c < 100.0):
        return 1e12
    if not (-20.0 < a0 < 20.0):
        return 1e12
    n = t.size
    two_pi_y = 2.0 * math.pi / 365.25
    T_safe = T if T > 1.0 else 1.0
    log_const = math.log(alpha - 1.0) + (alpha - 1.0) * math.log(c)
    lam_at = np.empty(n)
    for i in prange(n):
        nu_i = math.exp(a0 + a1 * (t[i] / T_safe)
                         + a2 * math.sin(two_pi_y * t[i])
                         + a3 * math.cos(two_pi_y * t[i]))
        s = 0.0
        for j in range(i):
            u = t[i] - t[j]
            if u > 0.0:
                log_d = log_const - alpha * math.log(u + c)
                s += math.exp(log_d)
        lam_at[i] = nu_i + eta * s
    log_sum = 0.0
    for i in range(n):
        if not math.isfinite(lam_at[i]) or lam_at[i] <= 0.0:
            return 1e12
        log_sum += math.log(lam_at[i])
    g_int = 0.0
    for k in range(grid.size - 1):
        g_int += 0.5 * (grid_vals[k] + grid_vals[k + 1]) * (grid[k + 1] - grid[k])
    integral = g_int
    for i in range(n):
        u = T - t[i]
        if u > 0.0:
            integral += eta * (1.0 - (c / (u + c)) ** (alpha - 1.0))
    return -(log_sum - integral)


# ── Dispatcher ──────────────────────────────────────────────────────


_SUPPORTED = {
    ("exponential", "constant"),
    ("exponential", "sinusoidal"),
    ("weibull", "constant"),
    ("weibull", "sinusoidal"),
    ("gamma", "constant"),
    ("lomax", "constant"),
    ("lomax", "sinusoidal"),
}

# Event count at/above which the constant-baseline Lomax likelihood is
# routed through the O(M*n) sum-of-exponentials engine instead of the
# exact O(n^2) kernel. Below it, O(n^2) is already cheap and is kept as
# the reference path. See soe_fit_lomax / hawkes_ll_soe.
_SOE_MIN_N = 1000


def _gamma_trunc_cutoff(alpha, beta):
    """Lag past which the gamma kernel underflows to exactly zero --
    mirrors the C++ hawkes_ll_gamma_const_trunc cutoff. Used to decide
    whether the bit-exact truncation degrades to O(n^2) (slow decay),
    in which case the gamma hybrid takes over."""
    import math
    log_const = alpha * math.log(beta) - math.lgamma(alpha)
    if alpha > 1.0 + 1e-12:
        k = ((alpha - 1.0) * math.log(2.0 * (alpha - 1.0) / beta)
             - (alpha - 1.0))
        return 2.0 * (log_const + k + 745.2) / beta
    return max(1.0, (log_const + 745.2) / beta)


def has_jit_path(kernel: str, baseline: str) -> bool:
    if HAS_CORE and baseline == "constant" and kernel in (
            "exponential", "weibull", "lomax", "gamma"):
        return True
    if HAS_CORE and baseline == "sinusoidal" and kernel in (
            "exponential", "weibull", "lomax"):
        return True
    return HAS_NUMBA and (kernel, baseline) in _SUPPORTED


def _sin_grid(T: float, a0: float, a1: float, a2: float, a3: float
              ) -> tuple[np.ndarray, np.ndarray]:
    n_grid = max(64, int(T) + 1)
    grid = np.linspace(0.0, T, n_grid)
    two_pi_y = 2.0 * math.pi / 365.25
    T_safe = T if T > 1.0 else 1.0
    vals = np.exp(a0 + a1 * (grid / T_safe)
                   + a2 * np.sin(two_pi_y * grid)
                   + a3 * np.cos(two_pi_y * grid))
    return grid, vals


def neg_loglik_jit(theta: np.ndarray, t: np.ndarray, T: float,
                    kernel: str, baseline: str) -> float:
    """JIT-routed negative log-likelihood.  Caller must check has_jit_path()."""
    # v0.9.1: constant-baseline kernels run on the compiled C++ core
    # when available -- no numba needed for these paths.
    if HAS_CORE and baseline == "constant":
        t_c = np.ascontiguousarray(t, dtype=np.float64)
        if kernel == "exponential":
            a0, eta, beta = float(theta[0]), float(theta[1]), float(theta[2])
            if eta <= 1e-6 or eta >= 0.999 or beta <= 1e-6:
                return 1e12
            return _core_ext.hawkes_ll_exp_const(t_c, float(T), a0, eta, beta)
        if kernel == "weibull":
            a0, eta = float(theta[0]), float(theta[1])
            alpha, lam = float(theta[2]), float(theta[3])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1e-6 or lam <= 1e-6:
                return 1e12
            # the sliding-window form is bit-identical and sub-quadratic
            return _core_ext.hawkes_ll_weibull_const_trunc(
                t_c, float(T), a0, eta, alpha, lam)
        if kernel == "lomax":
            a0, eta = float(theta[0]), float(theta[1])
            alpha, c = float(theta[2]), float(theta[3])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1.001 or c <= 1e-6:
                return 1e12
            # large n: route through the O(M*n) SoE engine. The SoE
            # perturbs the likelihood by ~1e-12 (far below the
            # optimizer's tolerance), so the MLE is unchanged; the
            # exact O(n^2) kernel stays the reference path below the
            # crossover, where it is already cheap.
            if t_c.shape[0] >= _SOE_MIN_N:
                w, beta_soe, _ = soe_fit_lomax(alpha, c, float(T), tol=1e-8)
                return _core_ext.hawkes_ll_soe(
                    t_c, float(T), float(np.exp(a0)), eta, w, beta_soe)
            return _core_ext.hawkes_ll_lomax_const(
                t_c, float(T), a0, eta, alpha, c)
        if kernel == "gamma":
            a0, eta = float(theta[0]), float(theta[1])
            alpha, beta = float(theta[2]), float(theta[3])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1e-6 or beta <= 1e-6:
                return 1e12
            # slow decay + large n: the bit-exact truncation degrades to
            # O(n^2), so route through the O(n*w + M*n) gamma hybrid
            # (exact peak window + matrix-pencil SoE tail). The hybrid
            # perturbs the likelihood by ~1e-6, below the optimizer's
            # tolerance. Truncation stays the exact path everywhere else.
            if (alpha > 1.0 and t_c.shape[0] >= _SOE_MIN_N
                    and _gamma_trunc_cutoff(alpha, beta)
                        >= t_c[-1] - t_c[0]):
                u_split = 2.0 * (alpha - 1.0) / beta
                w, beta_soe, _ = soe_fit_gamma_tail(alpha, beta, u_split)
                return _core_ext.hawkes_ll_gamma_hybrid(
                    t_c, float(T), a0, eta, alpha, beta, u_split,
                    w, beta_soe)
            # the sliding-window form is bit-identical and sub-quadratic
            return _core_ext.hawkes_ll_gamma_const_trunc(
                t_c, float(T), a0, eta, alpha, beta)
    if HAS_CORE and baseline == "sinusoidal" and kernel in (
            "exponential", "weibull", "lomax"):
        a0, a1, a2, a3 = (float(theta[0]), float(theta[1]),
                          float(theta[2]), float(theta[3]))
        eta = float(theta[4])
        grid, grid_vals = _sin_grid(T, a0, a1, a2, a3)
        t_c = np.ascontiguousarray(t, dtype=np.float64)
        g_c = np.ascontiguousarray(grid, dtype=np.float64)
        gv_c = np.ascontiguousarray(grid_vals, dtype=np.float64)
        if kernel == "exponential":
            beta = float(theta[5])
            if eta <= 1e-6 or eta >= 0.999 or beta <= 1e-6:
                return 1e12
            return _core_ext.hawkes_ll_exp_sin(
                t_c, float(T), a0, a1, a2, a3, eta, beta, g_c, gv_c)
        if kernel == "weibull":
            alpha, lam = float(theta[5]), float(theta[6])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1e-6 or lam <= 1e-6:
                return 1e12
            return _core_ext.hawkes_ll_weibull_sin(
                t_c, float(T), a0, a1, a2, a3, eta, alpha, lam, g_c, gv_c)
        if kernel == "lomax":
            alpha, c = float(theta[5]), float(theta[6])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1.001 or c <= 1e-6:
                return 1e12
            return _core_ext.hawkes_ll_lomax_sin(
                t_c, float(T), a0, a1, a2, a3, eta, alpha, c, g_c, gv_c)
    if not HAS_NUMBA:
        raise RuntimeError("neg_loglik_jit called without Numba available")

    if baseline == "constant":
        a0 = float(theta[0])
        eta = float(theta[1])
        if kernel == "exponential":
            beta = float(theta[2])
            if eta <= 1e-6 or eta >= 0.999 or beta <= 1e-6:
                return 1e12
            return _ll_exp_const(t, T, a0, eta, beta)
        if kernel == "weibull":
            alpha, lam = float(theta[2]), float(theta[3])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1e-6 or lam <= 1e-6:
                return 1e12
            return _ll_weibull_const(t, T, a0, eta, alpha, lam)
        if kernel == "gamma":
            alpha, beta = float(theta[2]), float(theta[3])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1e-6 or beta <= 1e-6:
                return 1e12
            return _ll_gamma_const(t, T, a0, eta, alpha, beta)
        if kernel == "lomax":
            alpha, c = float(theta[2]), float(theta[3])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1.001 or c <= 1e-6:
                return 1e12
            return _ll_lomax_const(t, T, a0, eta, alpha, c)
    elif baseline == "sinusoidal":
        a0, a1, a2, a3 = (float(theta[0]), float(theta[1]),
                            float(theta[2]), float(theta[3]))
        eta = float(theta[4])
        grid, grid_vals = _sin_grid(T, a0, a1, a2, a3)
        if kernel == "exponential":
            beta = float(theta[5])
            if eta <= 1e-6 or eta >= 0.999 or beta <= 1e-6:
                return 1e12
            return _ll_exp_sin(t, T, a0, a1, a2, a3, eta, beta, grid, grid_vals)
        if kernel == "weibull":
            alpha, lam = float(theta[5]), float(theta[6])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1e-6 or lam <= 1e-6:
                return 1e12
            return _ll_weibull_sin(t, T, a0, a1, a2, a3, eta, alpha, lam,
                                    grid, grid_vals)
        if kernel == "lomax":
            alpha, c = float(theta[5]), float(theta[6])
            if eta <= 1e-6 or eta >= 0.999 or alpha <= 1.001 or c <= 1e-6:
                return 1e12
            return _ll_lomax_sin(t, T, a0, a1, a2, a3, eta, alpha, c,
                                  grid, grid_vals)

    raise ValueError(f"no JIT path for kernel={kernel}, baseline={baseline}")


# ── User-callback bridge ────────────────────────────────────────────
#
# "Bring your own kernel": when morie's built-in triggering kernels
# misspecify a process, a user supplies their own. A numba @cfunc JITs
# the Python kernel to a native function pointer that morie's C++
# O(n^2) loop calls directly -- GIL-free, no per-call interpreter
# overhead. numba is an optional dependency: the morie[callbacks] extra.


def _as_cfunc(fn):
    """Return a numba @cfunc for *fn* (returned unchanged if already one).

    The returned object owns the compiled native code -- the caller
    must keep a reference to it while the native pointer is in use.
    Compiling a plain Python callable needs the morie[callbacks] extra.
    """
    if hasattr(fn, "address"):  # already a numba CFunc
        return fn
    try:
        from numba import cfunc, types
    except ImportError as exc:  # pragma: no cover -- callbacks extra absent
        raise ImportError(
            "hawkes_loglik_custom needs numba to compile a plain Python "
            "kernel -- install morie[callbacks], or pass a pre-built "
            "numba @cfunc.") from exc
    return cfunc(types.float64(types.float64))(fn)


def hawkes_loglik_custom(t, T, nu, eta, kernel, kernel_integral):
    """Hawkes negative log-likelihood with a user-supplied triggering kernel.

    The "bring your own kernel" path: supply your own triggering kernel
    when the built-in exponential / Weibull / Lomax / gamma kernels
    misspecify your process.

    Parameters
    ----------
    t : array
        Sorted event times.
    T : float
        Observation horizon.
    nu : float
        Constant baseline intensity (must be > 0).
    eta : float
        Branching ratio.
    kernel : callable or numba CFunc
        The triggering kernel ``g(dt)``. A numba ``@cfunc`` -- decorated
        ``@cfunc(numba.types.float64(numba.types.float64))`` -- is
        called natively inside morie's C++ O(n^2) loop, GIL-free. A
        plain Python callable is compiled to a ``@cfunc`` on the fly
        (needs the ``morie[callbacks]`` extra).
    kernel_integral : callable or numba CFunc
        ``G(u)`` = the integral of ``g`` over ``[0, u]`` -- the
        compensator term.

    Returns
    -------
    float
        The negative log-likelihood (a large sentinel if infeasible).

    Examples
    --------
    >>> import math
    >>> from numba import cfunc, types
    >>> @cfunc(types.float64(types.float64))
    ... def g(u):
    ...     return 1.5 * math.exp(-1.5 * u)
    >>> @cfunc(types.float64(types.float64))
    ... def G(u):
    ...     return 1.0 - math.exp(-1.5 * u)
    >>> hawkes_loglik_custom(t, T, nu=0.4, eta=0.3,
    ...                      kernel=g, kernel_integral=G)  # doctest: +SKIP
    """
    if not HAS_CORE:
        raise RuntimeError(
            "hawkes_loglik_custom requires the compiled morie C++ core "
            "(morie._core), which is not available in this install.")
    # Hold the CFunc objects in locals -- they own the compiled native
    # code, and the synchronous C++ call below dereferences their
    # addresses.
    g_cf = _as_cfunc(kernel)
    G_cf = _as_cfunc(kernel_integral)
    return _core_ext.hawkes_ll_custom(
        np.ascontiguousarray(t, dtype=np.float64), float(T),
        float(nu), float(eta), int(g_cf.address), int(G_cf.address))


# --- sum-of-exponentials (SoE) fit, task #73 -------------------------------
#
# A non-exponential triggering kernel makes the Hawkes likelihood O(n^2).
# If the kernel is written as a sum of M exponentials, each component
# regains the O(n) Markovian recursion, so the likelihood is O(M*n) via
# the C++ engine ``_core.hawkes_ll_soe``.
#
# For the Lomax (power-law) kernel the SoE is not a fitted approximation
# at all: the Lomax density is completely monotone, so by Bernstein's
# theorem it is *exactly* a continuous mixture of exponentials, and the
# SoE is just a quadrature of that integral (see soe_fit_lomax).
#
# The gamma kernel with shape alpha > 1 rises to a peak at
# (alpha-1)/beta, so it is NOT completely monotone and cannot be written
# as a non-negative mixture of exponentials at all. That case needs a
# hybrid (exact peak window + SoE tail) and is tracked separately.


def soe_fit_lomax(alpha, c, horizon, *, tol=1.0e-7, m_max=256):
    """Sum-of-exponentials representation of the Lomax triggering kernel,
    accurate over the lag range ``u`` in ``[0, horizon]``.

    The Lomax density ``g(u) = (alpha-1)/c * (1 + u/c)**(-alpha)`` is
    completely monotone, so by Bernstein's theorem it is *exactly* the
    Laplace transform of a non-negative mixing measure:

        g(u) = integral_0^inf  rho(s) e**(-s u) ds,
        rho(s) = (alpha-1) c**(alpha-1) / Gamma(alpha)
                 * s**(alpha-1) e**(-s c).

    A finite SoE has an exponential tail and so can never match the
    power-law tail as ``u -> inf``; but the Hawkes log-likelihood only
    evaluates inter-event lags up to the observation horizon, so the
    quadrature is targeted at ``[0, horizon]`` alone. Substituting
    ``s = exp(v)`` makes the integrand analytic and doubly-decaying in
    ``v``, for which the trapezoidal rule converges *geometrically*
    (Trefethen & Weideman, SIAM Review 2014): the node count grows only
    logarithmically in both the timescale range and the inverse
    tolerance.

    The decay-rate grid spans ``s_min ~ 1/horizon`` (the slowest mode
    resolvable over the data window) up to ``s_max ~ 40/c`` (past which
    ``rho`` is negligible -- ``e**(-s c)`` has decayed). ``M`` is not
    hand-tuned: it is doubled until the *measured* maximum relative
    error of the SoE against the exact kernel on ``[0, horizon]`` falls
    to ``tol`` (or ``m_max`` is hit).

    Returns ``(w, beta, max_rel_err)`` -- the float64 arrays for
    ``_core.hawkes_ll_soe`` plus the achieved, verified error bound.
    """
    import math

    if alpha <= 1.0:
        raise ValueError("Lomax triggering kernel requires alpha > 1")
    if c <= 0.0:
        raise ValueError("Lomax scale c must be positive")
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")

    s_min = 1.0e-7 / horizon
    s_max = max(40.0 / c, 1.0e3 * s_min)
    log_smin, log_smax = math.log(s_min), math.log(s_max)
    log_const = (math.log(alpha - 1.0) + (alpha - 1.0) * math.log(c)
                 - math.lgamma(alpha))

    # exact kernel on a check grid over [0, horizon] -- the SoE error is
    # measured directly against it, so M is data-driven, not guessed.
    u_chk = np.concatenate(([0.0],
                            np.geomspace(horizon * 1e-5, horizon, 400)))
    g_chk = np.exp(math.log(alpha - 1.0) - math.log(c)
                   - alpha * np.log1p(u_chk / c))

    w = beta = None
    err = float("inf")
    M = 32
    while True:
        v = np.linspace(log_smin, log_smax, M)
        s = np.exp(v)
        dv = v[1] - v[0]
        trap = np.full(M, dv)
        trap[0] *= 0.5
        trap[-1] *= 0.5
        rho = np.exp(log_const + (alpha - 1.0) * v - s * c)
        w = rho * s * trap                   # ds = s dv on the log grid
        beta = s
        g_soe = (w[None, :] * np.exp(-np.outer(u_chk, s))).sum(axis=1)
        err = float(np.max(np.abs(g_soe - g_chk) / g_chk))
        if err <= tol or M >= m_max:
            break
        M *= 2

    return (np.ascontiguousarray(w, dtype=np.float64),
            np.ascontiguousarray(beta, dtype=np.float64),
            err)


# --- matrix-pencil exponential fitter, task #73 (gamma hybrid) -------------
#
# The gamma kernel (shape alpha > 1) is not completely monotone and has
# no clean Bernstein mixing measure, so its tail cannot be quadratured
# the way the Lomax kernel is. Instead the tail is fitted empirically:
# the matrix-pencil method (Hua & Sarkar) recovers the exponential
# modes of uniformly-sampled data directly, by an eigen-decomposition
# -- no fixed grid of rates, so none of the collinearity that wrecked
# the earlier least-squares attempt.
#
# A completely monotone function's exponential modes are all real; if
# the pencil returns COMPLEX poles, that is a diagnostic -- the sample
# window still covers non-monotone (near-peak) curvature -- not modes
# to keep. The caller responds by moving the window, never by feeding
# oscillatory modes into the real-only SoE engine.


def _soe_fit_matrix_pencil(y, dt, *, order=None, rank_tol=1.0e-9):
    """Matrix-pencil (Hua-Sarkar) fit of uniformly-sampled data to a
    sum of exponentials:  y[k] ~= sum_m residue[m] * exp(-beta[m]*k*dt).

    y         uniformly-spaced samples y[k], k = 0 .. N-1
    dt        sample spacing
    order     model order M (number of exponentials); if None, taken
              from the numerical rank of the Hankel matrix
    rank_tol  singular values below rank_tol * sigma_max are dropped

    Returns (beta, residue) as complex arrays. A real decaying mode
    gives real beta > 0; a complex entry is an oscillatory mode and
    signals the fit window is not yet in the monotone tail.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 4:
        raise ValueError("matrix pencil needs at least 4 samples")

    pencil = n // 2                       # pencil parameter L
    rows = n - pencil
    hankel = np.empty((rows, pencil + 1))
    for i in range(rows):
        hankel[i] = y[i:i + pencil + 1]

    _, sv, vh = np.linalg.svd(hankel, full_matrices=False)
    if order is None:
        order = int(np.count_nonzero(sv > rank_tol * sv[0]))
    order = max(1, min(order, pencil))

    v = vh[:order].conj().T               # (pencil+1) x order
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])

    k = np.arange(n)
    vander = z[None, :] ** k[:, None]     # n x order
    residue, *_ = np.linalg.lstsq(vander, y, rcond=None)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = -np.log(z) / dt
    return beta, residue


def soe_fit_gamma_tail(alpha, beta, u_split, *, span=20.0, n_samples=240):
    """Matrix-pencil SoE fit of the gamma kernel's tail, for u >= u_split.

    The gamma triggering kernel
    ``g(u) = (beta**alpha / Gamma(alpha)) * u**(alpha-1) * exp(-beta*u)``
    is not completely monotone for shape ``alpha > 1``, so it has no
    Bernstein quadrature; the tail is fitted empirically instead. The
    kernel is sampled uniformly on ``[u_split, u_split + span/beta]``
    and the matrix pencil (``_soe_fit_matrix_pencil``) recovers the
    exponential modes.

    The fit is returned in the SHIFTED convention -- the modes describe
    ``g(u_split + s) = sum_m w[m] * exp(-beta_soe[m] * s)`` for ``s >=
    0`` -- which the hybrid engine's graduation recursion consumes
    directly, with no ``exp(beta*u_split)`` rescaling, so the weights
    stay well scaled.

    Modes are complex: real poles plus, for ``alpha`` above ~2.5,
    complex-conjugate pairs (an accurate damped-oscillatory mode, not
    an artefact -- see hawkes_ll_soe_cplx). Every ``Re(beta_soe)`` must
    be > 0 (decaying); a non-decaying mode raises ValueError.

    Returns ``(w, beta_soe, max_rel_err)`` -- complex128 arrays plus the
    measured relative error of the SoE against the exact tail.
    """
    import math

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("gamma kernel requires alpha > 0 and beta > 0")
    if u_split <= 0.0:
        raise ValueError("u_split must be positive")

    u = np.linspace(u_split, u_split + span / beta, n_samples)
    dt = u[1] - u[0]
    log_const = alpha * math.log(beta) - math.lgamma(alpha)
    g = np.exp(log_const + (alpha - 1.0) * np.log(u) - beta * u)

    # rank_tol 1e-13 (vs the fitter's conservative 1e-9 default) keeps
    # more singular values -> ~12 modes -> tail relative error ~1e-6;
    # the gamma tail's slowly-varying u**(alpha-1) factor needs them.
    scale = g[0]
    pole_beta, pole_res = _soe_fit_matrix_pencil(
        g / scale, dt, rank_tol=1.0e-13)
    w = pole_res * scale

    if np.any(pole_beta.real <= 0.0):
        raise ValueError("matrix-pencil fit produced a non-decaying mode")

    s = u - u_split
    g_soe = (w[None, :] * np.exp(-pole_beta[None, :] * s[:, None])
             ).sum(axis=1).real
    err = float(np.max(np.abs(g_soe - g) / g))
    return (np.ascontiguousarray(w, dtype=np.complex128),
            np.ascontiguousarray(pole_beta, dtype=np.complex128),
            err)
