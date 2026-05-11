"""Tests for morie.fn.eproc — empirical process."""

import numpy as np
import pytest

from morie.fn.eproc import eproc


def test_basic_output():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = eproc(x)
    assert "process" in result
    assert "ecdf" in result
    assert result["n"] == 200


def test_process_at_zero_mean():
    x = np.linspace(-3, 3, 1000)
    result = eproc(x)
    assert np.max(np.abs(result["process"])) < 10.0


def test_custom_cdf():
    from scipy.stats import uniform
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, 500)
    result = eproc(x, cdf_func=uniform.cdf)
    assert result["n"] == 500


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        eproc(np.array([]))


def test_eval_points():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    pts = np.array([-1.0, 0.0, 1.0])
    result = eproc(x, eval_points=pts)
    assert len(result["process"]) == 3
