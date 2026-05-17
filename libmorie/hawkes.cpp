// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- nanobind (Python) binding for the Hawkes likelihood.
//
// Thin adapter: unwraps the numpy array and calls the binding-agnostic
// core in morie_core.hpp. The same core function is bound for R via
// Rcpp.

#include "hawkes.h"
#include "morie_core.hpp"

#include <cstdint>

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

double hawkes_ll_weibull_const_trunc(Vec t, double T, double a0, double eta,
                                     double alpha, double lam) {
    return morie::core::hawkes_ll_weibull_const_trunc(
        t.data(), t.shape(0), T, a0, eta, alpha, lam);
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

// User-callback bridge: g_addr / G_addr are native function-pointer
// addresses (e.g. from numba @cfunc) for the triggering kernel and its
// integral. They are cast back to plain function pointers and called
// inside the C++ O(n^2) loop, GIL-free.
double hawkes_ll_custom(Vec t, double T, double nu, double eta,
                        std::uintptr_t g_addr, std::uintptr_t G_addr) {
    auto g = reinterpret_cast<morie::core::HawkesKernelFn>(g_addr);
    auto G = reinterpret_cast<morie::core::HawkesKernelFn>(G_addr);
    return morie::core::hawkes_ll_custom(t.data(), t.shape(0), T, nu, eta,
                                         g, G);
}

double hawkes_ll_soe(Vec t, double T, double nu, double eta, Vec w,
                     Vec beta) {
    return morie::core::hawkes_ll_soe(t.data(), t.shape(0), T, nu, eta,
                                      w.data(), beta.data(), w.shape(0));
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
          "constant baseline. Exact O(n^2). Returns 1e12 for an "
          "infeasible parameter vector.");
    m.def("hawkes_ll_weibull_const_trunc", &hawkes_ll_weibull_const_trunc,
          "t"_a, "T"_a, "a0"_a, "eta"_a, "alpha"_a, "lam"_a,
          "Sliding-window O(n*w) form of hawkes_ll_weibull_const -- "
          "bit-identical to the O(n^2) version (the truncated terms "
          "underflow to exactly zero).");
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
    m.def("hawkes_ll_custom", &hawkes_ll_custom, "t"_a, "T"_a, "nu"_a,
          "eta"_a, "g_addr"_a, "G_addr"_a,
          "Hawkes negative log-likelihood with a user-supplied triggering "
          "kernel. g_addr / G_addr are native function-pointer addresses "
          "(from numba @cfunc) for the kernel g(dt) and its integral "
          "G(u) = integral_0^u g.");
    m.def("hawkes_ll_soe", &hawkes_ll_soe, "t"_a, "T"_a, "nu"_a, "eta"_a,
          "w"_a, "beta"_a,
          "Hawkes negative log-likelihood with a sum-of-exponentials "
          "triggering kernel g(u) = sum_m w[m]*exp(-beta[m]*u). O(M*n) "
          "via M parallel exponential recursions.");
}
