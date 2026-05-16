# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoke tests for the compiled C++ core (``morie._core``).

Phase 0 of the v0.9.1 backend port: these prove the nanobind extension
is built, importable, and that the numpy bridge works. Real numeric
kernel tests land alongside later phases.

Requires the compiled extension -- runs under an installed or editable
(scikit-build-core) install, not a bare PYTHONPATH=src checkout.
"""
import array

import pytest

core = pytest.importorskip("morie._core")


def test_core_version():
    assert core.core_version() == "0.9.1-dev"


def test_add():
    assert core.add(2.0, 3.0) == 5.0
    assert core.add(-1.5, 1.5) == 0.0


def test_sum1d_buffer_protocol():
    assert core.sum1d(array.array("d", [1.0, 2.0, 3.0, 4.0])) == 10.0


def test_sum1d_numpy():
    np = pytest.importorskip("numpy")
    x = np.arange(100, dtype=np.float64)
    assert core.sum1d(x) == 4950.0
