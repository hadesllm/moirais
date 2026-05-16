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
// copy at r-package/morie/src/morie_core.hpp -- keep the two in sync.

#pragma once

#include <cmath>
#include <cstddef>

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

}  // namespace morie::core
