"""Tests for morie.fn.emprc — Empirical process (centered, scaled)."""

import numpy as np
import pytest

from morie.fn.emprc import emprc, EmpiricalProcessResult


@pytest.fixture()
def normal_sample():
    return np.random.default_rng(42).standard_normal(300)


def test_returns_result_type(normal_sample):
    result = emprc(normal_sample)
    assert isinstance(result, EmpiricalProcessResult)


def test_grid_length(normal_sample):
    result = emprc(normal_sample, n_grid=100)
    assert len(result.t_grid) == 100
    assert len(result.process) == 100


def test_sup_norm_positive(normal_sample):
    result = emprc(normal_sample)
    assert result.sup_norm > 0


def test_sup_norm_matches_process(normal_sample):
    result = emprc(normal_sample)
    assert np.isclose(result.sup_norm, np.max(np.abs(result.process)))


def test_process_bounded_for_normal():
    rng = np.random.default_rng(99)
    x = rng.standard_normal(1000)
    result = emprc(x)
    assert result.sup_norm < 10.0


def test_custom_cdf():
    from scipy.stats import uniform
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, size=200)
    result = emprc(x, cdf_fn=uniform.cdf)
    assert result.sup_norm < 5.0


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        emprc(np.array([]))


def test_bad_ngrid_raises():
    with pytest.raises(ValueError, match="n_grid"):
        emprc(np.array([1.0, 2.0]), n_grid=1)
