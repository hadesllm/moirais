# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the numba @cfunc -> C++ user-callback bridge (task #71).

A user-supplied triggering kernel, JIT-compiled to a native function
pointer, is called inside morie's C++ Hawkes loop. Requires the built
extension and numba (the morie[callbacks] extra)."""
import math

import numpy as np
import pytest

core = pytest.importorskip("morie._core")
numba = pytest.importorskip("numba")
from numba import cfunc, types

from morie.tps_hawkes_jit import _ll_exp_const, hawkes_loglik_custom


def _event_times(n, rate, seed):
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.exponential(1.0 / rate, size=n))


def test_custom_kernel_reproduces_builtin_exponential():
    # A user-supplied exponential kernel g(u) = beta*exp(-beta*u) with
    # integral G(u) = 1 - exp(-beta*u) must reproduce the built-in
    # exponential-kernel likelihood (which uses the O(n) recursion;
    # the custom path uses the O(n^2) direct sum -- same value).
    @cfunc(types.float64(types.float64))
    def g(u):
        return 1.5 * math.exp(-1.5 * u)

    @cfunc(types.float64(types.float64))
    def big_g(u):
        return 1.0 - math.exp(-1.5 * u)

    t = _event_times(300, rate=2.0, seed=11)
    T = float(t[-1]) + 1.0
    a0, eta = -1.0, 0.4
    got = hawkes_loglik_custom(t, T, math.exp(a0), eta, g, big_g)
    ref = float(_ll_exp_const(t, T, a0, eta, 1.5))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_custom_kernel_accepts_plain_python_callable():
    # A plain Python function is compiled to a @cfunc on the fly.
    def g(u):
        return 2.0 * math.exp(-2.0 * u)

    def big_g(u):
        return 1.0 - math.exp(-2.0 * u)

    t = _event_times(150, rate=1.5, seed=7)
    T = float(t[-1]) + 1.0
    got = hawkes_loglik_custom(t, T, 0.5, 0.3, g, big_g)
    ref = float(_ll_exp_const(t, T, math.log(0.5), 0.3, 2.0))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_core_hawkes_ll_custom_exposed():
    assert hasattr(core, "hawkes_ll_custom")
