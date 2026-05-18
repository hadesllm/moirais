// morie_fast.cpp -- Rcpp bindings for the R side of morie.
//
// Since v0.9.1 these functions are thin Rcpp adapters over the shared
// C++ numeric core in morie_core.h -- the SAME header the Python
// package binds via nanobind. The arithmetic is no longer duplicated
// per language: R and Python now call into one source of truth, which
// eliminates the Python<->R parity bug class by construction.
//
// Compile via R's standard mechanism (R CMD INSTALL). Without a C++
// toolchain at install time, R falls back to the pure-R kernels in
// R/_fast.R.
//
// morie_core.h is a vendored copy; the canonical file is
// libmorie/morie_core.hpp in the morie repository root. The R copy
// uses the .h extension so R CMD check accepts it as a src/ header.

#include <Rcpp.h>

#include <cstddef>

#include "morie_core.h"

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

// --- Hawkes negative log-likelihoods (constant baseline) -----------------
//
// Thin Rcpp adapters over the shared core -- the same functions the
// Python package binds via nanobind. The Weibull and gamma forms use
// the bit-identical sliding-window (sub-quadratic) variants. Each
// returns 1e12 for an infeasible parameter vector.

// [[Rcpp::export]]
double morie_hawkes_ll_exp_const_cpp(NumericVector t, double T, double a0,
                                     double eta, double beta) {
    return morie::core::hawkes_ll_exp_const(t.begin(), len(t), T, a0, eta,
                                            beta);
}

// [[Rcpp::export]]
double morie_hawkes_ll_weibull_const_cpp(NumericVector t, double T, double a0,
                                         double eta, double alpha,
                                         double lam) {
    return morie::core::hawkes_ll_weibull_const_trunc(t.begin(), len(t), T,
                                                      a0, eta, alpha, lam);
}

// [[Rcpp::export]]
double morie_hawkes_ll_lomax_const_cpp(NumericVector t, double T, double a0,
                                       double eta, double alpha, double c) {
    return morie::core::hawkes_ll_lomax_const(t.begin(), len(t), T, a0, eta,
                                              alpha, c);
}

// [[Rcpp::export]]
double morie_hawkes_ll_gamma_const_cpp(NumericVector t, double T, double a0,
                                       double eta, double alpha,
                                       double beta) {
    return morie::core::hawkes_ll_gamma_const_trunc(t.begin(), len(t), T, a0,
                                                    eta, alpha, beta);
}
