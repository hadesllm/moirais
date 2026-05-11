"""Tests for morie.fn.donsk — Donsker class membership test."""

import numpy as np
import pytest

from morie.fn.donsk import donsk, DonskerResult


@pytest.fixture()
def normal_sample():
    return np.random.default_rng(42).standard_normal(100)


def test_returns_result_type(normal_sample):
    result = donsk(normal_sample, n_boot=50, seed=1)
    assert isinstance(result, DonskerResult)


def test_is_donsker_for_normal(normal_sample):
    result = donsk(normal_sample, n_boot=200, seed=42)
    assert result.is_donsker is True


def test_p_value_range(normal_sample):
    result = donsk(normal_sample, n_boot=100, seed=7)
    assert 0.0 <= result.p_value <= 1.0


def test_observed_sup_positive(normal_sample):
    result = donsk(normal_sample, n_boot=50, seed=1)
    assert result.observed_sup > 0


def test_bootstrap_quantile_positive(normal_sample):
    result = donsk(normal_sample, n_boot=50, seed=1)
    assert result.bootstrap_quantile > 0


def test_n_stored(normal_sample):
    result = donsk(normal_sample, n_boot=50, seed=1)
    assert result.n == 100
    assert result.n_boot == 50


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        donsk(np.array([]))


def test_bad_nboot_raises():
    with pytest.raises(ValueError, match="n_boot"):
        donsk(np.array([1.0, 2.0]), n_boot=0)
