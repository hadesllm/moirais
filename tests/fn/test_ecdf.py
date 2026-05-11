"""Tests for morie.fn.ecdf — Empirical CDF with DKW confidence bands."""

import numpy as np
import pytest

from morie.fn.ecdf import ecdf, ECDFResult


@pytest.fixture()
def normal_sample():
    return np.random.default_rng(42).standard_normal(500)


def test_returns_result_type(normal_sample):
    result = ecdf(normal_sample)
    assert isinstance(result, ECDFResult)


def test_ecdf_monotone(normal_sample):
    result = ecdf(normal_sample)
    assert np.all(np.diff(result.ecdf_vals) >= 0)


def test_ecdf_range(normal_sample):
    result = ecdf(normal_sample)
    assert result.ecdf_vals[0] > 0
    assert result.ecdf_vals[-1] == 1.0


def test_bands_contain_ecdf(normal_sample):
    result = ecdf(normal_sample)
    assert np.all(result.lower <= result.ecdf_vals)
    assert np.all(result.upper >= result.ecdf_vals)


def test_bands_clipped_01(normal_sample):
    result = ecdf(normal_sample)
    assert np.all(result.lower >= 0.0)
    assert np.all(result.upper <= 1.0)


def test_epsilon_decreases_with_n():
    rng = np.random.default_rng(7)
    r100 = ecdf(rng.standard_normal(100))
    r1000 = ecdf(rng.standard_normal(1000))
    assert r1000.epsilon < r100.epsilon


def test_n_stored(normal_sample):
    result = ecdf(normal_sample)
    assert result.n == 500


def test_alpha_stored():
    x = np.array([1.0, 2.0, 3.0])
    result = ecdf(x, alpha=0.10)
    assert result.alpha == 0.10


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ecdf(np.array([]))


def test_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        ecdf(np.array([1.0, 2.0]), alpha=0.0)
