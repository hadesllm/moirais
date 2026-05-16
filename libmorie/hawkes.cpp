// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- nanobind (Python) binding for the Hawkes likelihood.
//
// Thin adapter: unwraps the numpy array and calls the binding-agnostic
// core in morie_core.hpp. The same core function is bound for R via
// Rcpp.

#include "hawkes.h"
#include "morie_core.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;

double hawkes_ll_exp_const(Vec t, double T, double a0, double eta,
                           double beta) {
    return morie::core::hawkes_ll_exp_const(t.data(), t.shape(0), T, a0,
                                            eta, beta);
}

double hawkes_ll_weibull_const(Vec t, double T, double a0, double eta,
                               double alpha, double lam) {
    return morie::core::hawkes_ll_weibull_const(t.data(), t.shape(0), T, a0,
                                                eta, alpha, lam);
}

double hawkes_ll_lomax_const(Vec t, double T, double a0, double eta,
                             double alpha, double c) {
    return morie::core::hawkes_ll_lomax_const(t.data(), t.shape(0), T, a0,
                                              eta, alpha, c);
}

double hawkes_ll_gamma_const(Vec t, double T, double a0, double eta,
                             double alpha, double beta) {
    return morie::core::hawkes_ll_gamma_const(t.data(), t.shape(0), T, a0,
                                              eta, alpha, beta);
}

}  // namespace

void register_hawkes(nb::module_ &m) {
    m.def("hawkes_ll_exp_const", &hawkes_ll_exp_const, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "beta"_a,
          "Hawkes negative log-likelihood -- exponential triggering "
          "kernel, constant baseline. Returns 1e12 for an infeasible "
          "parameter vector.");
    m.def("hawkes_ll_weibull_const", &hawkes_ll_weibull_const, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "alpha"_a, "lam"_a,
          "Hawkes negative log-likelihood -- Weibull triggering kernel, "
          "constant baseline. Returns 1e12 for an infeasible parameter "
          "vector.");
    m.def("hawkes_ll_lomax_const", &hawkes_ll_lomax_const, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "alpha"_a, "c"_a,
          "Hawkes negative log-likelihood -- Lomax (power-law) triggering "
          "kernel, constant baseline. The caller enforces alpha > 1 and "
          "c > 0.");
    m.def("hawkes_ll_gamma_const", &hawkes_ll_gamma_const, "t"_a, "T"_a,
          "a0"_a, "eta"_a, "alpha"_a, "beta"_a,
          "Hawkes negative log-likelihood -- gamma triggering kernel, "
          "constant baseline. Returns 1e12 for an infeasible parameter "
          "vector.");
}
