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


def has_jit_path(kernel: str, baseline: str) -> bool:
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
