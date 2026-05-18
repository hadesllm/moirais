// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- the morie C++ numeric core (binding-agnostic).
//
// This header is THE single numeric source of truth for morie. It is
// header-only and depends on nothing but the C++ standard library --
// no nanobind, no Rcpp, no numpy. Every function takes plain raw
// pointers + sizes so it can be bound, unchanged, by:
//
//   * nanobind  -> Python  (libmorie/kernels.cpp, libmorie/hawkes.cpp)
//   * Rcpp      -> R       (r-package/morie/src/rcpp_morie.cpp)
//
// Because both languages call into the SAME compiled arithmetic, the
// Python<->R parity bug class (e.g. row-major vs column-major, subtly
// divergent reimplementations) is eliminated by construction.
//
// CANONICAL COPY: libmorie/morie_core.hpp. The R package vendors a
// copy at r-package/morie/src/morie_core.h -- keep the two in sync.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

namespace morie::core {

inline const double kPi = 3.14159265358979323846;
inline const double kInvSqrt2Pi = 1.0 / std::sqrt(2.0 * kPi);
inline const double kLogSqrt2Pi = 0.5 * std::log(2.0 * kPi);

// Sentinel for an infeasible parameter vector (Hawkes likelihood).
inline const double kBig = 1e12;

// --- summary statistics ------------------------------------------------------

inline double mean(const double *a, std::size_t n) {
    if (n == 0) return std::nan("");
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += a[i];
    return s / static_cast<double>(n);
}

inline double variance(const double *a, std::size_t n, int ddof) {
    if (static_cast<long long>(n) - ddof <= 0) return std::nan("");
    const double m = mean(a, n);
    double sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = a[i] - m;
        sq += d * d;
    }
    return sq / (static_cast<double>(n) - static_cast<double>(ddof));
}

inline double stddev(const double *a, std::size_t n, int ddof) {
    return std::sqrt(variance(a, n, ddof));
}

inline double cor_pearson(const double *x, const double *y, std::size_t n) {
    if (n < 2) return std::nan("");
    double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = x[i], b = y[i];
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    const double dn = static_cast<double>(n);
    const double num = dn * sxy - sx * sy;
    const double den_sq = (dn * sxx - sx * sx) * (dn * syy - sy * sy);
    if (den_sq <= 0.0) return std::nan("");
    return num / std::sqrt(den_sq);
}

inline double euclid_dist(const double *a, const double *b, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

// --- array-valued kernels (write into a caller-provided buffer) --------------

inline void normal_pdf(const double *x, std::size_t n, double mean_,
                       double sd, double *out) {
    const double inv = 1.0 / sd;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = (x[i] - mean_) * inv;
        out[i] = inv * kInvSqrt2Pi * std::exp(-0.5 * z * z);
    }
}

inline void normal_logpdf(const double *x, std::size_t n, double mean_,
                          double sd, double *out) {
    const double inv = 1.0 / sd;
    const double base = -std::log(sd) - kLogSqrt2Pi;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = (x[i] - mean_) * inv;
        out[i] = base - 0.5 * z * z;
    }
}

inline void trimmed_ipw_weights(const double *treat, const double *propensity,
                                std::size_t n, double trim_lo,
                                double trim_hi, double *out) {
    for (std::size_t i = 0; i < n; ++i) {
        double e = propensity[i];
        if (e < trim_lo) {
            e = trim_lo;
        } else if (e > trim_hi) {
            e = trim_hi;
        }
        out[i] = (treat[i] == 1.0) ? (1.0 / e) : (1.0 / (1.0 - e));
    }
}

// Bootstrap-replicate means: B resamples of size n drawn with
// replacement from `a`, each replicate's mean written to out[b].
// Uses std::mt19937_64 seeded with `seed` -- fully reproducible for a
// given seed. (There is intentionally no pure-numpy equivalent: a
// different RNG would silently change the replicates.)
inline void bootstrap_mean(const double *a, std::size_t n, std::size_t B,
                           unsigned long long seed, double *out) {
    if (n == 0) {
        for (std::size_t b = 0; b < B; ++b) out[b] = std::nan("");
        return;
    }
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> idx(0, n - 1);
    for (std::size_t b = 0; b < B; ++b) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            s += a[idx(rng)];
        }
        out[b] = s / static_cast<double>(n);
    }
}

// --- Hawkes-process likelihood ----------------------------------------------

// Negative log-likelihood: exponential triggering kernel, constant
// baseline. The O(n) recursion A_i = exp(-beta*dt)*(A_{i-1}+beta) is
// genuinely sequential. Returns kBig for an infeasible parameter set.
inline double hawkes_ll_exp_const(const double *t, std::size_t n, double T,
                                  double a0, double eta, double beta) {
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(0.05 < beta && beta < 30.0)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;

    const double nu = std::exp(a0);
    if (!std::isfinite(nu) || nu <= 0.0) return kBig;

    double log_sum = 0.0;
    double A = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double lam_i;
        if (i == 0) {
            lam_i = nu;
        } else {
            const double dt = t[i] - t[i - 1];
            A = std::exp(-beta * dt) * (A + beta);
            lam_i = nu + eta * A;
        }
        if (!std::isfinite(lam_i) || lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        integral += eta * (1.0 - std::exp(-beta * (T - t[i])));
    }
    if (!std::isfinite(log_sum) || !std::isfinite(integral)) return kBig;

    return -(log_sum - integral);
}

// Negative log-likelihood: Weibull triggering kernel, constant
// baseline. The Weibull kernel is not memoryless, so there is no O(n)
// recursion -- each event sums over all prior events (O(n^2)).
// Returns kBig for an infeasible parameter set.
inline double hawkes_ll_weibull_const(const double *t, std::size_t n, double T,
                                      double a0, double eta, double alpha,
                                      double lam) {
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(0.05 < alpha && alpha < 20.0)) return kBig;
    if (!(1e-3 < lam && lam < 1e3)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;

    const double nu = std::exp(a0);

    double log_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            const double x = (t[i] - t[j]) / lam;
            if (x > 1e-12) {
                const double z = std::pow(x, alpha);
                if (z < 700.0) {
                    s += (alpha / lam) * std::pow(x, alpha - 1.0) *
                         std::exp(-z);
                }
            }
        }
        const double lam_at = nu + eta * s;
        if (lam_at <= 0.0) return kBig;
        log_sum += std::log(lam_at);
    }

    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = T - t[i];
        if (u > 0.0) {
            const double x = u / lam;
            integral += eta * (1.0 - std::exp(-std::pow(x, alpha)));
        }
    }
    return -(log_sum - integral);
}

// Negative log-likelihood: Lomax (Omori-type power-law) triggering
// kernel, constant baseline. Like the Weibull kernel this is not
// memoryless -- exact O(n^2). The caller is responsible for the
// parameter bounds (alpha > 1 so log(alpha-1) is finite, c > 0).
inline double hawkes_ll_lomax_const(const double *t, std::size_t n, double T,
                                    double a0, double eta, double alpha,
                                    double c) {
    const double nu = std::exp(a0);
    const double log_const =
        std::log(alpha - 1.0) + (alpha - 1.0) * std::log(c);

    double log_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            const double u = t[i] - t[j];
            const double log_d = log_const - alpha * std::log(u + c);
            s += std::exp(log_d);
        }
        const double lam_i = nu + eta * s;
        if (lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = T - t[i];
        if (u > 0.0) {
            integral += eta * (1.0 - std::pow(c / (u + c), alpha - 1.0));
        }
    }
    return -(log_sum - integral);
}

// Regularized lower incomplete gamma P(a, x): series for x < a+1,
// continued fraction (Lentz) otherwise (Numerical Recipes 6.2). Used
// by the gamma-kernel Hawkes compensator integral.
inline double gamma_cdf_regularized(double a, double x) {
    if (x <= 0.0) return 0.0;
    const double gln = std::lgamma(a);
    if (x < a + 1.0) {
        double ap = a;
        double sum = 1.0 / a;
        double delta = sum;
        for (int i = 0; i < 200; ++i) {
            ap += 1.0;
            delta *= x / ap;
            sum += delta;
            if (std::fabs(delta) < std::fabs(sum) * 1e-12) break;
        }
        return sum * std::exp(-x + a * std::log(x) - gln);
    }
    double b = x + 1.0 - a;
    double c = 1.0 / 1e-30;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= 200; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < 1e-30) d = 1e-30;
        c = b + an / c;
        if (std::fabs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::fabs(delta - 1.0) < 1e-12) break;
    }
    return 1.0 - std::exp(-x + a * std::log(x) - gln) * h;
}

// Negative log-likelihood: gamma triggering kernel, constant baseline.
// Non-Markovian -- exact O(n^2). The compensator integral uses the
// regularized incomplete gamma above. Returns kBig when infeasible.
inline double hawkes_ll_gamma_const(const double *t, std::size_t n, double T,
                                    double a0, double eta, double alpha,
                                    double beta) {
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(0.05 < alpha && alpha < 20.0)) return kBig;
    if (!(0.05 < beta && beta < 30.0)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;

    const double nu = std::exp(a0);
    const double log_const = alpha * std::log(beta) - std::lgamma(alpha);

    double log_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            const double u = t[i] - t[j];
            if (u > 1e-300) {
                const double log_d =
                    log_const + (alpha - 1.0) * std::log(u) - beta * u;
                s += std::exp(log_d);
            }
        }
        const double lam_i = nu + eta * s;
        if (lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        integral += eta * gamma_cdf_regularized(alpha, beta * (T - t[i]));
    }
    return -(log_sum - integral);
}

// Negative log-likelihood: exponential triggering kernel, sinusoidal
// (time-varying) baseline. The baseline integral is supplied as a
// pre-built grid (grid / grid_vals, length n_grid) and integrated by
// the trapezoidal rule; the caller builds the grid. The O(n) recursion
// on the exponential kernel still applies. Returns kBig if infeasible.
inline double hawkes_ll_exp_sin(const double *t, std::size_t n, double T,
                                double a0, double a1, double a2, double a3,
                                double eta, double beta, const double *grid,
                                const double *grid_vals, std::size_t n_grid) {
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(0.05 < beta && beta < 30.0)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;

    const double two_pi_y = 2.0 * kPi / 365.25;
    const double T_safe = (T > 1.0) ? T : 1.0;

    double log_sum = 0.0;
    double A = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double nu_i = std::exp(a0 + a1 * (t[i] / T_safe) +
                                     a2 * std::sin(two_pi_y * t[i]) +
                                     a3 * std::cos(two_pi_y * t[i]));
        double lam_i;
        if (i == 0) {
            lam_i = nu_i;
        } else {
            const double dt = t[i] - t[i - 1];
            A = std::exp(-beta * dt) * (A + beta);
            lam_i = nu_i + eta * A;
        }
        if (lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    // baseline integral: trapezoidal rule over the supplied grid
    double g_int = 0.0;
    for (std::size_t k = 0; k + 1 < n_grid; ++k) {
        g_int += 0.5 * (grid_vals[k] + grid_vals[k + 1]) *
                 (grid[k + 1] - grid[k]);
    }
    double integral = g_int;
    for (std::size_t i = 0; i < n; ++i) {
        integral += eta * (1.0 - std::exp(-beta * (T - t[i])));
    }
    return -(log_sum - integral);
}

// Negative log-likelihood: Weibull triggering kernel, sinusoidal
// baseline. Combines the O(n^2) Weibull inner sum with the trapezoid
// baseline integral over the caller-supplied grid. Returns kBig when
// infeasible.
inline double hawkes_ll_weibull_sin(const double *t, std::size_t n, double T,
                                    double a0, double a1, double a2, double a3,
                                    double eta, double alpha, double lam,
                                    const double *grid,
                                    const double *grid_vals,
                                    std::size_t n_grid) {
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(0.05 < alpha && alpha < 20.0)) return kBig;
    if (!(1e-3 < lam && lam < 1e3)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;

    const double two_pi_y = 2.0 * kPi / 365.25;
    const double T_safe = (T > 1.0) ? T : 1.0;

    double log_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double nu_i = std::exp(a0 + a1 * (t[i] / T_safe) +
                                     a2 * std::sin(two_pi_y * t[i]) +
                                     a3 * std::cos(two_pi_y * t[i]));
        double s = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            const double x = (t[i] - t[j]) / lam;
            if (x > 1e-12) {
                const double z = std::pow(x, alpha);
                if (z < 700.0) {
                    s += (alpha / lam) * std::pow(x, alpha - 1.0) *
                         std::exp(-z);
                }
            }
        }
        const double lam_i = nu_i + eta * s;
        if (!std::isfinite(lam_i) || lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    double g_int = 0.0;
    for (std::size_t k = 0; k + 1 < n_grid; ++k) {
        g_int += 0.5 * (grid_vals[k] + grid_vals[k + 1]) *
                 (grid[k + 1] - grid[k]);
    }
    double integral = g_int;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = T - t[i];
        if (u > 0.0) {
            const double x = u / lam;
            integral += eta * (1.0 - std::exp(-std::pow(x, alpha)));
        }
    }
    return -(log_sum - integral);
}

// Negative log-likelihood: Lomax (Omori-type power-law) triggering
// kernel, sinusoidal baseline. O(n^2) inner sum + the trapezoid
// baseline integral. The kernel enforces its own parameter bounds
// (alpha in (1.05, 30), c in (1e-4, 100)). Returns kBig if infeasible.
inline double hawkes_ll_lomax_sin(const double *t, std::size_t n, double T,
                                  double a0, double a1, double a2, double a3,
                                  double eta, double alpha, double c,
                                  const double *grid, const double *grid_vals,
                                  std::size_t n_grid) {
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(1.05 < alpha && alpha < 30.0)) return kBig;
    if (!(1e-4 < c && c < 100.0)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;

    const double two_pi_y = 2.0 * kPi / 365.25;
    const double T_safe = (T > 1.0) ? T : 1.0;
    const double log_const =
        std::log(alpha - 1.0) + (alpha - 1.0) * std::log(c);

    double log_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double nu_i = std::exp(a0 + a1 * (t[i] / T_safe) +
                                     a2 * std::sin(two_pi_y * t[i]) +
                                     a3 * std::cos(two_pi_y * t[i]));
        double s = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            const double u = t[i] - t[j];
            if (u > 0.0) {
                const double log_d = log_const - alpha * std::log(u + c);
                s += std::exp(log_d);
            }
        }
        const double lam_i = nu_i + eta * s;
        if (!std::isfinite(lam_i) || lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    double g_int = 0.0;
    for (std::size_t k = 0; k + 1 < n_grid; ++k) {
        g_int += 0.5 * (grid_vals[k] + grid_vals[k + 1]) *
                 (grid[k + 1] - grid[k]);
    }
    double integral = g_int;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = T - t[i];
        if (u > 0.0) {
            integral += eta * (1.0 - std::pow(c / (u + c), alpha - 1.0));
        }
    }
    return -(log_sum - integral);
}

// --- user-callback bridge ----------------------------------------------------
//
// A C-ABI function pointer double(double). The numba @cfunc bridge JITs
// a user's Python kernel into exactly this, and the C++ loop below
// calls it natively -- no Python interpreter, no GIL -- inside the
// O(n^2) hot loop.
using HawkesKernelFn = double (*)(double);

// Generic Hawkes negative log-likelihood with a USER-supplied
// triggering kernel: g(dt) is the kernel and G(u) = integral_0^u g.
// Both are plain function pointers, so the O(n^2) excitation sum and
// the compensator call user code at native speed. Returns kBig when
// infeasible.
inline double hawkes_ll_custom(const double *t, std::size_t n, double T,
                               double nu, double eta, HawkesKernelFn g,
                               HawkesKernelFn G) {
    if (!(nu > 0.0) || !std::isfinite(nu)) return kBig;

    double log_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            s += g(t[i] - t[j]);
        }
        const double lam_i = nu + eta * s;
        if (!std::isfinite(lam_i) || lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        integral += eta * G(T - t[i]);
    }
    if (!std::isfinite(log_sum) || !std::isfinite(integral)) return kBig;
    return -(log_sum - integral);
}

// --- sub-quadratic Weibull (task #72) ---------------------------------------
//
// Truncated / sliding-window form of hawkes_ll_weibull_const. Beyond
// u = lam * 700^(1/alpha) the kernel's exp(-(u/lam)^alpha) underflows
// to exactly 0 -- those terms contribute nothing, and the exact O(n^2)
// version already skips them via its `z < 700` guard. So cutting the
// inner loop there is EXACT, bit-for-bit identical to the O(n^2)
// result, not an approximation -- it cannot bias the MLE.
//
// Event times are sorted, so the lower bound advances monotonically:
// a two-pointer window gives O(n*w), w = events within the cutoff.
// For a slowly-decaying kernel (small alpha) w -> n and it degrades
// gracefully to O(n^2), still exact.
inline double hawkes_ll_weibull_const_trunc(const double *t, std::size_t n,
                                            double T, double a0, double eta,
                                            double alpha, double lam) {
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(0.05 < alpha && alpha < 20.0)) return kBig;
    if (!(1e-3 < lam && lam < 1e3)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;

    const double nu = std::exp(a0);
    const double cutoff = lam * std::pow(700.0, 1.0 / alpha);

    double log_sum = 0.0;
    std::size_t lo = 0;
    for (std::size_t i = 0; i < n; ++i) {
        while (lo < i && (t[i] - t[lo]) > cutoff) ++lo;
        double s = 0.0;
        for (std::size_t j = lo; j < i; ++j) {
            const double x = (t[i] - t[j]) / lam;
            if (x > 1e-12) {
                const double z = std::pow(x, alpha);
                if (z < 700.0) {
                    s += (alpha / lam) * std::pow(x, alpha - 1.0) *
                         std::exp(-z);
                }
            }
        }
        const double lam_at = nu + eta * s;
        if (lam_at <= 0.0) return kBig;
        log_sum += std::log(lam_at);
    }

    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = T - t[i];
        if (u > 0.0) {
            const double x = u / lam;
            integral += eta * (1.0 - std::exp(-std::pow(x, alpha)));
        }
    }
    return -(log_sum - integral);
}

// Truncated / sliding-window form of hawkes_ll_gamma_const. The gamma
// kernel's log-density  log_d = log_const + (alpha-1)*log(u) - beta*u
// falls monotonically past the peak; beyond the cutoff below it is
// < -745, so exp(log_d) underflows to exactly 0. The exact O(n^2)
// version already adds those zeros, so cutting the inner loop there is
// bit-for-bit identical -- not an approximation, cannot bias the MLE.
//
// No closed form solves log_d = -745 (the mixed log(u)-beta*u term),
// so the cutoff uses a guaranteed upper bound on (alpha-1)*log(u):
// for alpha > 1, (alpha-1)log(u) <= (beta/2)u + K with K the bound's
// maximum; for alpha <= 1 the term is <= 0 once u >= 1. The cutoff is
// thus a little wider than the true underflow point -- conservative
// but always correct. For a slowly decaying kernel the window -> n and
// it degrades gracefully to exact O(n^2).
inline double hawkes_ll_gamma_const_trunc(const double *t, std::size_t n,
                                          double T, double a0, double eta,
                                          double alpha, double beta) {
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(0.05 < alpha && alpha < 20.0)) return kBig;
    if (!(0.05 < beta && beta < 30.0)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;

    const double nu = std::exp(a0);
    const double log_const = alpha * std::log(beta) - std::lgamma(alpha);

    double cutoff;
    if (alpha > 1.0 + 1e-12) {
        const double K = (alpha - 1.0) *
                             std::log(2.0 * (alpha - 1.0) / beta) -
                         (alpha - 1.0);
        cutoff = 2.0 * (log_const + K + 745.2) / beta;
    } else {
        cutoff = (log_const + 745.2) / beta;
        if (cutoff < 1.0) cutoff = 1.0;
    }

    double log_sum = 0.0;
    std::size_t lo = 0;
    for (std::size_t i = 0; i < n; ++i) {
        while (lo < i && (t[i] - t[lo]) > cutoff) ++lo;
        double s = 0.0;
        for (std::size_t j = lo; j < i; ++j) {
            const double u = t[i] - t[j];
            if (u > 1e-300) {
                const double log_d =
                    log_const + (alpha - 1.0) * std::log(u) - beta * u;
                s += std::exp(log_d);
            }
        }
        const double lam_i = nu + eta * s;
        if (lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        integral += eta * gamma_cdf_regularized(alpha, beta * (T - t[i]));
    }
    return -(log_sum - integral);
}

// --- sum-of-exponentials (SoE) Hawkes likelihood (task #73) ------------------
//
// Hawkes negative log-likelihood with a sum-of-exponentials triggering
// kernel:  g(u) = sum_m  w[m] * exp(-beta[m] * u).
//
// Each exponential component is memoryless, so it carries its own O(n)
// recursion  A_m,i = exp(-beta_m * dt) * (1 + A_m,{i-1}).  Running M
// such recursions in parallel evaluates the likelihood in O(M*n) --
// the sub-quadratic engine for any kernel (Weibull / gamma / Lomax)
// once it has been fitted to an SoE form.
//
// With M = 1 and w = beta = {b} this reduces exactly (to rounding) to
// the exponential-kernel likelihood hawkes_ll_exp_const.
inline double hawkes_ll_soe(const double *t, std::size_t n, double T,
                            double nu, double eta, const double *w,
                            const double *beta, std::size_t M) {
    if (!(nu > 0.0) || !std::isfinite(nu)) return kBig;

    std::vector<double> A(M, 0.0);  // one recursion state per component
    double log_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double excite = 0.0;
        if (i > 0) {
            const double dt = t[i] - t[i - 1];
            for (std::size_t m = 0; m < M; ++m) {
                A[m] = std::exp(-beta[m] * dt) * (1.0 + A[m]);
                excite += w[m] * A[m];
            }
        }
        const double lam_i = nu + eta * excite;
        if (!std::isfinite(lam_i) || lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    // compensator: integral of g over [0, T-t_i] is, per component,
    // (w_m / beta_m) * (1 - exp(-beta_m * (T - t_i))).
    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = T - t[i];
        for (std::size_t m = 0; m < M; ++m) {
            integral += eta * (w[m] / beta[m]) *
                        (1.0 - std::exp(-beta[m] * u));
        }
    }
    if (!std::isfinite(log_sum) || !std::isfinite(integral)) return kBig;
    return -(log_sum - integral);
}

// Complex-pole SoE Hawkes likelihood (task #73, gamma hybrid).
//
// Same O(M*n) recursion as hawkes_ll_soe, but the decay rates beta and
// weights w are complex. The matrix-pencil fit of a gamma tail returns
// real poles plus complex-conjugate pairs; a conjugate pair is the
// real damped oscillation  2*Re(w*exp(-beta*u)), so the excitation and
// the compensator are accumulated in complex arithmetic and the real
// part is taken. With purely real (w, beta) this is identical to
// hawkes_ll_soe.
//
// The caller must pass conjugate poles in matching pairs, so the
// imaginary parts cancel and lambda is real; Re() then only discards
// rounding noise. Re(beta) > 0 keeps |exp(-beta*dt)| < 1, so the
// recursion is stable -- the fitter guarantees this.
inline double hawkes_ll_soe_cplx(const double *t, std::size_t n, double T,
                                 double nu, double eta,
                                 const std::complex<double> *w,
                                 const std::complex<double> *beta,
                                 std::size_t M) {
    if (!(nu > 0.0) || !std::isfinite(nu)) return kBig;

    std::vector<std::complex<double>> A(M, std::complex<double>(0.0, 0.0));
    double log_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double excite = 0.0;
        if (i > 0) {
            const double dt = t[i] - t[i - 1];
            std::complex<double> acc(0.0, 0.0);
            for (std::size_t m = 0; m < M; ++m) {
                A[m] = std::exp(-beta[m] * dt) * (1.0 + A[m]);
                acc += w[m] * A[m];
            }
            excite = acc.real();
        }
        const double lam_i = nu + eta * excite;
        if (!std::isfinite(lam_i) || lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = T - t[i];
        std::complex<double> acc(0.0, 0.0);
        for (std::size_t m = 0; m < M; ++m) {
            acc += (w[m] / beta[m]) *
                   (1.0 - std::exp(-beta[m] * u));
        }
        integral += eta * acc.real();
    }
    if (!std::isfinite(log_sum) || !std::isfinite(integral)) return kBig;
    return -(log_sum - integral);
}

// Hybrid gamma-kernel Hawkes likelihood (task #73).
//
// The gamma kernel (shape alpha > 1) is not completely monotone, so it
// has no global sum-of-exponentials form. This splits the lag axis at
// u_split: lags in [0, u_split] (the rising/peak region) use the EXACT
// kernel via a sliding window; lags beyond u_split use the SoE fitted
// by soe_fit_gamma_tail (w_soe, beta_soe -- complex, shifted so the
// modes describe g(u_split + s)).
//
// Events sorted => two monotone pointers. j_grad counts events whose
// lag has crossed u_split; as it advances, each graduating event is
// folded into the SoE state S_m (a "graduation" recursion). Events in
// [j_grad, i) are still inside the exact window. Cost O(n*w + M*n),
// w = events within u_split. The compensator splits the same way: the
// exact part is the regularized incomplete gamma, the tail part the
// closed-form SoE integral.
inline double hawkes_ll_gamma_hybrid(
    const double *t, std::size_t n, double T, double a0, double eta,
    double alpha, double beta, double u_split,
    const std::complex<double> *w_soe,
    const std::complex<double> *beta_soe, std::size_t M) {
    if (!(1e-6 < eta && eta < 0.999)) return kBig;
    if (!(0.05 < alpha && alpha < 20.0)) return kBig;
    if (!(0.05 < beta && beta < 30.0)) return kBig;
    if (!(-20.0 < a0 && a0 < 20.0)) return kBig;
    if (!(u_split > 0.0)) return kBig;

    const double nu = std::exp(a0);
    const double log_const = alpha * std::log(beta) - std::lgamma(alpha);

    std::vector<std::complex<double>> S(M, std::complex<double>(0.0, 0.0));
    double log_sum = 0.0;
    std::size_t j_grad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) {
            const double dt = t[i] - t[i - 1];
            for (std::size_t m = 0; m < M; ++m)
                S[m] *= std::exp(-beta_soe[m] * dt);
        }
        // graduate events whose lag has just crossed u_split
        while (j_grad < i && (t[i] - t[j_grad]) > u_split) {
            const double s = t[i] - t[j_grad] - u_split;
            for (std::size_t m = 0; m < M; ++m)
                S[m] += std::exp(-beta_soe[m] * s);
            ++j_grad;
        }
        // tail excitation (SoE) ...
        std::complex<double> tail(0.0, 0.0);
        for (std::size_t m = 0; m < M; ++m) tail += w_soe[m] * S[m];
        double excite = tail.real();
        // ... plus window excitation (exact kernel)
        for (std::size_t j = j_grad; j < i; ++j) {
            const double u = t[i] - t[j];
            if (u > 1e-300) {
                const double log_d =
                    log_const + (alpha - 1.0) * std::log(u) - beta * u;
                excite += std::exp(log_d);
            }
        }
        const double lam_i = nu + eta * excite;
        if (!std::isfinite(lam_i) || lam_i <= 0.0) return kBig;
        log_sum += std::log(lam_i);
    }

    double integral = nu * T;
    for (std::size_t i = 0; i < n; ++i) {
        const double X = T - t[i];
        if (X <= 0.0) continue;
        if (X <= u_split) {
            integral += eta * gamma_cdf_regularized(alpha, beta * X);
        } else {
            integral += eta * gamma_cdf_regularized(alpha, beta * u_split);
            const double x_tail = X - u_split;
            std::complex<double> csum(0.0, 0.0);
            for (std::size_t m = 0; m < M; ++m)
                csum += (w_soe[m] / beta_soe[m]) *
                        (1.0 - std::exp(-beta_soe[m] * x_tail));
            integral += eta * csum.real();
        }
    }
    if (!std::isfinite(log_sum) || !std::isfinite(integral)) return kBig;
    return -(log_sum - integral);
}

}  // namespace morie::core
