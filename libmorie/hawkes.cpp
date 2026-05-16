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

double hawkes_ll_exp_sin(Vec t, double T, double a0, double a1, double a2,
                         double a3, double eta, double beta, Vec grid,
                         Vec grid_vals) {
    return morie::core::hawkes_ll_exp_sin(
        t.data(), t.shape(0), T, a0, a1, a2, a3, eta, beta, grid.data(),
        grid_vals.data(), grid.shape(0));
}

double hawkes_ll_weibull_sin(Vec t, double T, double a0, double a1, double a2,
                             double a3, double eta, double alpha, double lam,
                             Vec grid, Vec grid_vals) {
    return morie::core::hawkes_ll_weibull_sin(
        t.data(), t.shape(0), T, a0, a1, a2, a3, eta, alpha, lam,
        grid.data(), grid_vals.data(), grid.shape(0));
}

double hawkes_ll_lomax_sin(Vec t, double T, double a0, double a1, double a2,
                           double a3, double eta, double alpha, double c,
                           Vec grid, Vec grid_vals) {
    return morie::core::hawkes_ll_lomax_sin(
        t.data(), t.shape(0), T, a0, a1, a2, a3, eta, alpha, c,
        grid.data(), grid_vals.data(), grid.shape(0));
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
    m.def("hawkes_ll_exp_sin", &hawkes_ll_exp_sin, "t"_a, "T"_a, "a0"_a,
          "a1"_a, "a2"_a, "a3"_a, "eta"_a, "beta"_a, "grid"_a, "grid_vals"_a,
          "Hawkes negative log-likelihood -- exponential triggering "
          "kernel, sinusoidal baseline (trapezoid grid). Returns 1e12 "
          "for an infeasible parameter vector.");
    m.def("hawkes_ll_weibull_sin", &hawkes_ll_weibull_sin, "t"_a, "T"_a,
          "a0"_a, "a1"_a, "a2"_a, "a3"_a, "eta"_a, "alpha"_a, "lam"_a,
          "grid"_a, "grid_vals"_a,
          "Hawkes negative log-likelihood -- Weibull triggering kernel, "
          "sinusoidal baseline (trapezoid grid). Returns 1e12 for an "
          "infeasible parameter vector.");
    m.def("hawkes_ll_lomax_sin", &hawkes_ll_lomax_sin, "t"_a, "T"_a,
          "a0"_a, "a1"_a, "a2"_a, "a3"_a, "eta"_a, "alpha"_a, "c"_a,
          "grid"_a, "grid_vals"_a,
          "Hawkes negative log-likelihood -- Lomax (power-law) triggering "
          "kernel, sinusoidal baseline (trapezoid grid). Returns 1e12 for "
          "an infeasible parameter vector.");
}
