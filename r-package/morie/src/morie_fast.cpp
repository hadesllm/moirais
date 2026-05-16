// morie_fast.cpp -- Rcpp bindings for the R side of morie.
//
// Since v0.9.1 these functions are thin Rcpp adapters over the shared
// C++ numeric core in morie_core.hpp -- the SAME header the Python
// package binds via nanobind. The arithmetic is no longer duplicated
// per language: R and Python now call into one source of truth, which
// eliminates the Python<->R parity bug class by construction.
//
// Compile via R's standard mechanism (R CMD INSTALL). Without a C++
// toolchain at install time, R falls back to the pure-R kernels in
// R/_fast.R.
//
// morie_core.hpp is a vendored copy; the canonical file is
// libmorie/morie_core.hpp in the morie repository root.

#include <Rcpp.h>

#include <cstddef>

#include "morie_core.hpp"

using namespace Rcpp;

namespace {
inline std::size_t len(const NumericVector &v) {
    return static_cast<std::size_t>(v.size());
}
}  // namespace

// [[Rcpp::export]]
NumericVector morie_normal_pdf_cpp(NumericVector x, double mean, double sd) {
    if (sd <= 0.0) {
        Rcpp::stop("sd must be positive");
    }
    NumericVector out(x.size());
    morie::core::normal_pdf(x.begin(), len(x), mean, sd, out.begin());
    return out;
}

// [[Rcpp::export]]
double morie_mean_cpp(NumericVector x) {
    return morie::core::mean(x.begin(), len(x));
}

// [[Rcpp::export]]
double morie_var_cpp(NumericVector x, int ddof = 1) {
    return morie::core::variance(x.begin(), len(x), ddof);
}

// [[Rcpp::export]]
double morie_cor_pearson_cpp(NumericVector x, NumericVector y) {
    if (x.size() != y.size()) {
        return NA_REAL;
    }
    return morie::core::cor_pearson(x.begin(), y.begin(), len(x));
}
