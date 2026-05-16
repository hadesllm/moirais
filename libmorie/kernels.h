// SPDX-License-Identifier: AGPL-3.0-or-later
//
// libmorie kernels -- registration entry point.
#pragma once

#include <nanobind/nanobind.h>

// Binds the C++ numeric kernels into the morie._core module.
void register_kernels(nanobind::module_ &m);
