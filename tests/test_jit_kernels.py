# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity tests for the C++ numeric kernels (morie._core via morie._jit).

Phase 1 of the v0.9.1 backend port: each compiled-C++ kernel result is
checked against an independent pure-numpy reference. Requires the built
extension -- skipped on a bare PYTHONPATH=src checkout.
"""
import math

import numpy as np
import pytest

pytest.importorskip("morie._core")

from morie import _jit


@pytest.fixture
def rng():
    return np.random.RandomState(20260516)


def test_core_is_used():
    # in a built install the C++ core runs, not the numpy fallback
    assert _jit.is_jit_available() is True


def test_normal_pdf(rng):
    x = rng.normal(size=500)
    got = _jit.normal_pdf(x, 0.3, 1.7)
    inv = 1.0 / 1.7
    z = (x - 0.3) * inv
    ref = inv / math.sqrt(2 * math.pi) * np.exp(-0.5 * z * z)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-15)


def test_normal_logpdf(rng):
    x = rng.normal(size=500)
    got = _jit.normal_logpdf(x, -1.0, 2.5)
    ref = np.log(_jit.normal_pdf(x, -1.0, 2.5))
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_mean_var_std(rng):
    a = rng.normal(size=1000)
    assert np.isclose(_jit.mean_jit(a), np.mean(a), rtol=1e-12)
    assert np.isclose(_jit.var_jit(a, ddof=1), np.var(a, ddof=1), rtol=1e-10)
    assert np.isclose(_jit.var_jit(a, ddof=0), np.var(a, ddof=0), rtol=1e-10)
    assert np.isclose(_jit.std_jit(a, ddof=1), np.std(a, ddof=1), rtol=1e-10)


def test_cor_pearson(rng):
    x = rng.normal(size=400)
    y = 0.6 * x + rng.normal(size=400)
    assert np.isclose(_jit.cor_pearson_jit(x, y),
                      np.corrcoef(x, y)[0, 1], rtol=1e-9)


def test_euclid_dist(rng):
    a = rng.normal(size=300)
    b = rng.normal(size=300)
    assert np.isclose(_jit.euclid_dist_jit(a, b),
                      np.linalg.norm(a - b), rtol=1e-12)


def test_trimmed_ipw_weights(rng):
    treat = rng.randint(0, 2, size=200).astype(float)
    ps = rng.uniform(0.0, 1.0, size=200)
    got = _jit.trimmed_ipw_weights_jit(treat, ps, 0.05, 0.95)
    e = np.clip(ps, 0.05, 0.95)
    ref = np.where(treat == 1.0, 1.0 / e, 1.0 / (1.0 - e))
    assert np.allclose(got, ref, rtol=1e-12)


def test_edge_cases():
    assert math.isnan(_jit.mean_jit(np.array([])))
    assert math.isnan(_jit.var_jit(np.array([5.0]), ddof=1))  # n - ddof <= 0
    assert math.isnan(_jit.cor_pearson_jit(np.array([1.0]), np.array([2.0])))


def test_bootstrap_mean_reproducible(rng):
    # bootstrap stays pure-numpy; verify it is reproducible and sane
    a = rng.normal(loc=5.0, size=400)
    r1 = _jit.bootstrap_mean_jit(a, B=200, seed=7)
    r2 = _jit.bootstrap_mean_jit(a, B=200, seed=7)
    assert np.array_equal(r1, r2)
    assert r1.shape == (200,)
    assert abs(r1.mean() - a.mean()) < 0.2
