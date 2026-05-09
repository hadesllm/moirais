"""Tests for moirais.fn.glivn — Glivenko-Cantelli uniform convergence test."""

import numpy as np
import pytest

from moirais.fn.glivn import glivn, GlivenkoCantelliResult


@pytest.fixture()
def normal_sample():
    return np.random.default_rng(42).standard_normal(500)


def test_returns_result_type(normal_sample):
    result = glivn(normal_sample)
    assert isinstance(result, GlivenkoCantelliResult)


def test_converges_for_correct_cdf(normal_sample):
    result = glivn(normal_sample)
    assert result.converges is True


def test_does_not_converge_wrong_cdf():
    from scipy.stats import uniform
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    result = glivn(x, cdf_fn=uniform.cdf)
    assert result.converges is False


def test_sup_deviation_positive(normal_sample):
    result = glivn(normal_sample)
    assert result.sup_deviation > 0


def test_dkw_bound_positive(normal_sample):
    result = glivn(normal_sample)
    assert result.dkw_bound > 0


def test_bound_shrinks_with_n():
    rng = np.random.default_rng(7)
    r100 = glivn(rng.standard_normal(100))
    r1000 = glivn(rng.standard_normal(1000))
    assert r1000.dkw_bound < r100.dkw_bound


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        glivn(np.array([]))


def test_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        glivn(np.array([1.0, 2.0]), alpha=1.5)
