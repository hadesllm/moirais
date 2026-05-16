// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie kernels -- C++ ports of morie's numeric hot paths,
// formerly the numba-decorated functions in morie/_jit.py.
//
// Deterministic kernels only: results match the pure-numpy reference
// to within floating-point rounding. bootstrap_mean_jit is NOT ported
// here -- its np.random sequence is reproducible per seed, and a C++
// RNG would silently change that, so it stays in the Python layer
// pending a deliberate RNG decision (Phase 4).

#include "kernels.h"

#include <cmath>
#include <cstddef>

#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Computed once at load -- matches Python's 1/sqrt(2*pi) and
// 0.5*log(2*pi) (same double-precision sqrt/log).
const double kPi = 3.14159265358979323846;
const double kInvSqrt2Pi = 1.0 / std::sqrt(2.0 * kPi);
const double kLogSqrt2Pi = 0.5 * std::log(2.0 * kPi);

// A read-only, contiguous 1-D float64 array (the Python shim coerces
// every input to this layout before calling in).
using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;
using OutArray = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

// Allocate an owned float64 array that nanobind hands back to numpy;
// the capsule frees it when the numpy array is garbage-collected.
OutArray make_array(std::size_t n, double **out) {
    double *data = new double[n];
    *out = data;
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] static_cast<double *>(p);
    });
    return OutArray(data, {n}, owner);
}

OutArray normal_pdf(Vec x, double mean, double sd) {
    const std::size_t n = x.shape(0);
    double *out;
    OutArray arr = make_array(n, &out);
    const double inv_sigma = 1.0 / sd;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = (x(i) - mean) * inv_sigma;
        out[i] = inv_sigma * kInvSqrt2Pi * std::exp(-0.5 * z * z);
    }
    return arr;
}

OutArray normal_logpdf(Vec x, double mean, double sd) {
    const std::size_t n = x.shape(0);
    double *out;
    OutArray arr = make_array(n, &out);
    const double inv_sigma = 1.0 / sd;
    const double base = -std::log(sd) - kLogSqrt2Pi;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = (x(i) - mean) * inv_sigma;
        out[i] = base - 0.5 * z * z;
    }
    return arr;
}

double mean_jit(Vec arr) {
    const std::size_t n = arr.shape(0);
    if (n == 0) return std::nan("");
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += arr(i);
    return s / static_cast<double>(n);
}

double var_jit(Vec arr, int ddof) {
    const std::size_t n = arr.shape(0);
    if (static_cast<long long>(n) - ddof <= 0) return std::nan("");
    const double m = mean_jit(arr);
    double sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = arr(i) - m;
        sq += d * d;
    }
    return sq / (static_cast<double>(n) - static_cast<double>(ddof));
}

double std_jit(Vec arr, int ddof) { return std::sqrt(var_jit(arr, ddof)); }

double cor_pearson_jit(Vec x, Vec y) {
    const std::size_t n = x.shape(0);
    if (n != y.shape(0) || n < 2) return std::nan("");
    double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = x(i), b = y(i);
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

double euclid_dist_jit(Vec a, Vec b) {
    const std::size_t n = a.shape(0);
    if (n != b.shape(0)) return std::nan("");
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = a(i) - b(i);
        s += d * d;
    }
    return std::sqrt(s);
}

OutArray trimmed_ipw_weights_jit(Vec treat, Vec propensity, double trim_lo,
                                 double trim_hi) {
    const std::size_t n = treat.shape(0);
    double *out;
    OutArray arr = make_array(n, &out);
    for (std::size_t i = 0; i < n; ++i) {
        double e = propensity(i);
        if (e < trim_lo) {
            e = trim_lo;
        } else if (e > trim_hi) {
            e = trim_hi;
        }
        out[i] = (treat(i) == 1.0) ? (1.0 / e) : (1.0 / (1.0 - e));
    }
    return arr;
}

}  // namespace

void register_kernels(nb::module_ &m) {
    m.def("normal_pdf", &normal_pdf, "x"_a, "mean"_a, "sd"_a,
          "Normal PDF over a 1-D float64 array.");
    m.def("normal_logpdf", &normal_logpdf, "x"_a, "mean"_a, "sd"_a,
          "Normal log-density over a 1-D float64 array.");
    m.def("mean_jit", &mean_jit, "arr"_a, "Arithmetic mean of a 1-D array.");
    m.def("var_jit", &var_jit, "arr"_a, "ddof"_a = 1,
          "Sample variance with ddof (two-pass).");
    m.def("std_jit", &std_jit, "arr"_a, "ddof"_a = 1,
          "Sample standard deviation with ddof.");
    m.def("cor_pearson_jit", &cor_pearson_jit, "x"_a, "y"_a,
          "Pearson correlation coefficient.");
    m.def("euclid_dist_jit", &euclid_dist_jit, "a"_a, "b"_a,
          "Euclidean (L2) distance between two equal-length vectors.");
    m.def("trimmed_ipw_weights_jit", &trimmed_ipw_weights_jit, "treat"_a,
          "propensity"_a, "trim_lo"_a = 0.01, "trim_hi"_a = 0.99,
          "IPW weights with propensity-score clipping.");
}
