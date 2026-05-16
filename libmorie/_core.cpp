// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie -- the morie C++ numeric core.
//
// Phase 0 of the v0.9.1 backend port: this file is the
// proof-of-toolchain. It builds via scikit-build-core + CMake +
// nanobind into the importable extension module ``morie._core``.
//
// The functions here are deliberately trivial -- a scalar smoke and a
// numpy-array smoke -- their only job is to prove the build pipeline
// and the zero-copy numpy bridge that the real numeric kernels
// (Phase 1 onward) will rely on. The same compiled core is bound for
// R via Rcpp in Phase 3.

#include <cstddef>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "kernels.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Scalar smoke -- proves the binding layer end to end.
double add(double a, double b) { return a + b; }

// Array smoke -- proves the zero-copy numpy bridge. Real kernels
// (Phase 1+) take numpy arrays through exactly this interface.
double sum1d(nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x) {
    double total = 0.0;
    const std::size_t n = x.shape(0);
    for (std::size_t i = 0; i < n; ++i) {
        total += x(i);
    }
    return total;
}

// Identifies the compiled core (distinct from morie.__version__,
// which is the package version).
const char *core_version() { return "0.9.1-dev"; }

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "morie C++ numeric core (libmorie).";
    m.def("core_version", &core_version,
          "Version string of the compiled morie C++ core.");
    m.def("add", &add, "a"_a, "b"_a,
          "Add two doubles. Toolchain smoke test.");
    m.def("sum1d", &sum1d, "x"_a,
          "Sum a 1-D float64 array. numpy-bridge smoke test.");

    // Numeric kernels (ports of the former morie/_jit.py functions).
    register_kernels(m);
}
