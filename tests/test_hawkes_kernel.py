# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity tests for the C++ Hawkes likelihood kernel (Phase 2).

The compiled hawkes_ll_exp_const is checked against the reference
morie/tps_hawkes_jit.py::_ll_exp_const (same arithmetic). Requires the
built extension.
"""
import numpy as np
import pytest

core = pytest.importorskip("morie._core")

from morie.tps_hawkes_jit import _ll_exp_const


def _event_times(n, rate, seed):
    """Sorted event times from an exponential inter-arrival process."""
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.exponential(1.0 / rate, size=n))


@pytest.mark.parametrize("a0,eta,beta", [
    (-1.0, 0.30, 1.5),
    (0.5, 0.60, 0.8),
    (-2.0, 0.10, 5.0),
    (1.0, 0.85, 0.2),
])
def test_hawkes_ll_exp_const_parity(a0, eta, beta):
    t = _event_times(400, rate=2.0, seed=11)
    T = float(t[-1]) + 1.0
    got = core.hawkes_ll_exp_const(t, T, a0, eta, beta)
    ref = float(_ll_exp_const(t, T, a0, eta, beta))
    assert np.isclose(got, ref, rtol=1e-12, atol=1e-9)


def test_hawkes_ll_exp_const_infeasible_returns_sentinel():
    t = _event_times(50, rate=1.0, seed=3)
    T = float(t[-1]) + 1.0
    # eta outside (1e-6, 0.999)
    assert core.hawkes_ll_exp_const(t, T, 0.0, 1.5, 1.0) == 1e12
    # beta outside (0.05, 30.0)
    assert core.hawkes_ll_exp_const(t, T, 0.0, 0.3, 100.0) == 1e12
    # a0 outside (-20, 20)
    assert core.hawkes_ll_exp_const(t, T, 50.0, 0.3, 1.0) == 1e12


def test_neg_loglik_jit_routes_through_core():
    # neg_loglik_jit's exp/constant path uses the C++ core
    from morie.tps_hawkes_jit import has_jit_path, neg_loglik_jit
    assert has_jit_path("exponential", "constant") is True
    t = _event_times(200, rate=1.5, seed=8)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.4, 1.2])
    got = neg_loglik_jit(theta, t, T, "exponential", "constant")
    ref = float(_ll_exp_const(t, T, -1.0, 0.4, 1.2))
    assert np.isclose(got, ref, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("a0,eta,alpha,lam", [
    (-1.0, 0.30, 1.5, 2.0),
    (0.5, 0.60, 0.8, 5.0),
    (-2.0, 0.10, 3.0, 1.0),
])
def test_hawkes_ll_weibull_const_parity(a0, eta, alpha, lam):
    from morie.tps_hawkes_jit import _ll_weibull_const
    t = _event_times(250, rate=2.0, seed=13)
    T = float(t[-1]) + 1.0
    got = core.hawkes_ll_weibull_const(t, T, a0, eta, alpha, lam)
    ref = float(_ll_weibull_const(t, T, a0, eta, alpha, lam))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_hawkes_ll_weibull_const_infeasible_returns_sentinel():
    t = _event_times(40, rate=1.0, seed=5)
    T = float(t[-1]) + 1.0
    assert core.hawkes_ll_weibull_const(t, T, 0.0, 1.5, 1.5, 2.0) == 1e12
    assert core.hawkes_ll_weibull_const(t, T, 0.0, 0.3, 50.0, 2.0) == 1e12


def test_neg_loglik_jit_routes_weibull_through_core():
    from morie.tps_hawkes_jit import _ll_weibull_const, has_jit_path
    from morie.tps_hawkes_jit import neg_loglik_jit
    assert has_jit_path("weibull", "constant") is True
    t = _event_times(180, rate=1.5, seed=9)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.4, 1.5, 2.0])
    got = neg_loglik_jit(theta, t, T, "weibull", "constant")
    ref = float(_ll_weibull_const(t, T, -1.0, 0.4, 1.5, 2.0))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("a0,eta,alpha,c", [
    (-1.0, 0.30, 2.0, 1.0),
    (0.5, 0.60, 3.5, 0.5),
    (-2.0, 0.10, 1.5, 2.0),
])
def test_hawkes_ll_lomax_const_parity(a0, eta, alpha, c):
    from morie.tps_hawkes_jit import _ll_lomax_const
    t = _event_times(250, rate=2.0, seed=17)
    T = float(t[-1]) + 1.0
    got = core.hawkes_ll_lomax_const(t, T, a0, eta, alpha, c)
    ref = float(_ll_lomax_const(t, T, a0, eta, alpha, c))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_lomax_through_core():
    from morie.tps_hawkes_jit import _ll_lomax_const, has_jit_path
    from morie.tps_hawkes_jit import neg_loglik_jit
    assert has_jit_path("lomax", "constant") is True
    t = _event_times(180, rate=1.5, seed=21)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.4, 2.0, 1.0])
    got = neg_loglik_jit(theta, t, T, "lomax", "constant")
    ref = float(_ll_lomax_const(t, T, -1.0, 0.4, 2.0, 1.0))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)
    # alpha <= 1.001 -> the dispatch returns the infeasible sentinel
    bad = neg_loglik_jit(np.array([0.0, 0.3, 1.0, 1.0]), t, T,
                         "lomax", "constant")
    assert bad == 1e12


@pytest.mark.parametrize("a0,eta,alpha,beta", [
    (-1.0, 0.30, 2.0, 1.5),
    (0.5, 0.60, 0.8, 3.0),
    (-2.0, 0.10, 5.0, 0.5),
])
def test_hawkes_ll_gamma_const_parity(a0, eta, alpha, beta):
    from morie.tps_hawkes_jit import _ll_gamma_const
    t = _event_times(250, rate=2.0, seed=23)
    T = float(t[-1]) + 1.0
    got = core.hawkes_ll_gamma_const(t, T, a0, eta, alpha, beta)
    ref = float(_ll_gamma_const(t, T, a0, eta, alpha, beta))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_gamma_through_core():
    from morie.tps_hawkes_jit import _ll_gamma_const, has_jit_path
    from morie.tps_hawkes_jit import neg_loglik_jit
    assert has_jit_path("gamma", "constant") is True
    t = _event_times(180, rate=1.5, seed=27)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.4, 2.0, 1.5])
    got = neg_loglik_jit(theta, t, T, "gamma", "constant")
    ref = float(_ll_gamma_const(t, T, -1.0, 0.4, 2.0, 1.5))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)
    # alpha outside (0.05, 20) -> infeasible sentinel
    assert core.hawkes_ll_gamma_const(t, T, 0.0, 0.3, 50.0, 1.5) == 1e12


@pytest.mark.parametrize("a0,a1,a2,a3,eta,beta", [
    (-1.0, 0.2, 0.3, -0.1, 0.30, 1.5),
    (0.0, -0.5, 0.1, 0.4, 0.55, 0.8),
])
def test_hawkes_ll_exp_sin_parity(a0, a1, a2, a3, eta, beta):
    from morie.tps_hawkes_jit import _ll_exp_sin, _sin_grid
    t = _event_times(250, rate=2.0, seed=31)
    T = float(t[-1]) + 1.0
    grid, grid_vals = _sin_grid(T, a0, a1, a2, a3)
    got = core.hawkes_ll_exp_sin(t, T, a0, a1, a2, a3, eta, beta,
                                 grid, grid_vals)
    ref = float(_ll_exp_sin(t, T, a0, a1, a2, a3, eta, beta,
                            grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_exp_sin_through_core():
    from morie.tps_hawkes_jit import _ll_exp_sin, _sin_grid, has_jit_path
    from morie.tps_hawkes_jit import neg_loglik_jit
    assert has_jit_path("exponential", "sinusoidal") is True
    t = _event_times(180, rate=1.5, seed=33)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.2, 0.3, -0.1, 0.4, 1.2])
    got = neg_loglik_jit(theta, t, T, "exponential", "sinusoidal")
    grid, grid_vals = _sin_grid(T, -1.0, 0.2, 0.3, -0.1)
    ref = float(_ll_exp_sin(t, T, -1.0, 0.2, 0.3, -0.1, 0.4, 1.2,
                            grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("a0,a1,a2,a3,eta,alpha,lam", [
    (-1.0, 0.2, 0.3, -0.1, 0.30, 1.5, 2.0),
    (0.0, -0.5, 0.1, 0.4, 0.55, 0.8, 5.0),
])
def test_hawkes_ll_weibull_sin_parity(a0, a1, a2, a3, eta, alpha, lam):
    from morie.tps_hawkes_jit import _ll_weibull_sin, _sin_grid
    t = _event_times(250, rate=2.0, seed=37)
    T = float(t[-1]) + 1.0
    grid, grid_vals = _sin_grid(T, a0, a1, a2, a3)
    got = core.hawkes_ll_weibull_sin(t, T, a0, a1, a2, a3, eta, alpha, lam,
                                     grid, grid_vals)
    ref = float(_ll_weibull_sin(t, T, a0, a1, a2, a3, eta, alpha, lam,
                                grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_weibull_sin_through_core():
    from morie.tps_hawkes_jit import _ll_weibull_sin, _sin_grid
    from morie.tps_hawkes_jit import has_jit_path, neg_loglik_jit
    assert has_jit_path("weibull", "sinusoidal") is True
    t = _event_times(180, rate=1.5, seed=39)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.2, 0.3, -0.1, 0.4, 1.5, 2.0])
    got = neg_loglik_jit(theta, t, T, "weibull", "sinusoidal")
    grid, grid_vals = _sin_grid(T, -1.0, 0.2, 0.3, -0.1)
    ref = float(_ll_weibull_sin(t, T, -1.0, 0.2, 0.3, -0.1, 0.4, 1.5, 2.0,
                                grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("a0,a1,a2,a3,eta,alpha,c", [
    (-1.0, 0.2, 0.3, -0.1, 0.30, 2.0, 1.0),
    (0.0, -0.5, 0.1, 0.4, 0.55, 3.5, 0.5),
])
def test_hawkes_ll_lomax_sin_parity(a0, a1, a2, a3, eta, alpha, c):
    from morie.tps_hawkes_jit import _ll_lomax_sin, _sin_grid
    t = _event_times(250, rate=2.0, seed=41)
    T = float(t[-1]) + 1.0
    grid, grid_vals = _sin_grid(T, a0, a1, a2, a3)
    got = core.hawkes_ll_lomax_sin(t, T, a0, a1, a2, a3, eta, alpha, c,
                                   grid, grid_vals)
    ref = float(_ll_lomax_sin(t, T, a0, a1, a2, a3, eta, alpha, c,
                              grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)


def test_neg_loglik_jit_routes_lomax_sin_through_core():
    from morie.tps_hawkes_jit import _ll_lomax_sin, _sin_grid
    from morie.tps_hawkes_jit import has_jit_path, neg_loglik_jit
    assert has_jit_path("lomax", "sinusoidal") is True
    t = _event_times(180, rate=1.5, seed=43)
    T = float(t[-1]) + 1.0
    theta = np.array([-1.0, 0.2, 0.3, -0.1, 0.4, 2.0, 1.0])
    got = neg_loglik_jit(theta, t, T, "lomax", "sinusoidal")
    grid, grid_vals = _sin_grid(T, -1.0, 0.2, 0.3, -0.1)
    ref = float(_ll_lomax_sin(t, T, -1.0, 0.2, 0.3, -0.1, 0.4, 2.0, 1.0,
                              grid, grid_vals))
    assert np.isclose(got, ref, rtol=1e-9, atol=1e-6)
